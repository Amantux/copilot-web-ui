#!/usr/bin/env python3
"""copilot-web-ui — A ChatGPT-style web interface for GitHub Copilot CLI.

Spawns `copilot -p <prompt> --output-format json` per message and streams
the response to the browser over Server-Sent Events (SSE). Sessions persist
across messages via `--continue` (resumes the most recent session).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import AsyncIterator

from aiohttp import web


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
HTML_DIR = HERE / "html"

COPILOT_BIN = os.environ.get("COPILOT_BIN", "copilot")
HOST = os.environ.get("COPILOT_WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("COPILOT_WEB_PORT", "8765"))
WORKSPACE = Path(os.environ.get("COPILOT_WORKSPACE", str(HERE / "workspace"))).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

_ENV_PASSWORD = os.environ.get("COPILOT_WEB_PASSWORD", "").strip()
PASSWORD = _ENV_PASSWORD or secrets.token_urlsafe(16)
PASSWORD_AUTO = not _ENV_PASSWORD

SECRET = os.environ.get("COPILOT_WEB_SECRET") or secrets.token_hex(32)
COOKIE_NAME = "cwui_session"
SESSION_TTL = int(os.environ.get("COPILOT_WEB_SESSION_TTL", str(7 * 24 * 3600)))
COOKIE_SECURE = os.environ.get("COPILOT_WEB_COOKIE_SECURE", "auto").lower()

PUBLIC_PATHS = {"/", "/login", "/login.html", "/api/login", "/api/auth/status", "/healthz"}
PUBLIC_PREFIXES = ("/css/", "/js/", "/fonts/", "/img/")

# Trusted network CIDRs — requests from these ranges skip password auth entirely.
# Set COPILOT_TRUSTED_NETWORKS="" to disable. Default: 192.168.0.0/16
_TRUSTED_NETS_RAW = os.environ.get("COPILOT_TRUSTED_NETWORKS", "192.168.0.0/16")
TRUSTED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
for _cidr in _TRUSTED_NETS_RAW.split(","):
    _cidr = _cidr.strip()
    if _cidr:
        try:
            TRUSTED_NETWORKS.append(ipaddress.ip_network(_cidr, strict=False))
        except ValueError:
            print(f"  ⚠ Invalid CIDR in COPILOT_TRUSTED_NETWORKS: {_cidr!r}", file=sys.stderr)


def _is_trusted_ip(request: web.Request) -> bool:
    """Return True if the client IP is within a trusted network (no auth needed)."""
    if not TRUSTED_NETWORKS:
        return False
    raw = request.headers.get("X-Forwarded-For") or request.remote or ""
    client_ip = raw.split(",")[0].strip()
    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return any(addr in net for net in TRUSTED_NETWORKS)

# In-memory chat history per browser session id  { session_id: [{"role": "user"|"assistant", "content": str}] }
_chat_histories: dict[str, list[dict]] = {}

# Map browser session_id -> copilot session name (used with --name / --resume)
_copilot_sessions: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Auth helpers  (same approach as Copilot-Spawner)
# ---------------------------------------------------------------------------

def _sign(payload: str) -> str:
    mac = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def _make_token(ttl: int = SESSION_TTL) -> str:
    expires = int(time.time()) + ttl
    payload = f"v1.{expires}"
    return f"{payload}.{_sign(payload)}"


def _verify_token(token: str | None) -> bool:
    if not token:
        return False
    try:
        version, exp_s, sig = token.split(".")
    except ValueError:
        return False
    if version != "v1":
        return False
    try:
        expires = int(exp_s)
    except ValueError:
        return False
    if expires < int(time.time()):
        return False
    return hmac.compare_digest(sig, _sign(f"{version}.{exp_s}"))


def _is_authenticated(request: web.Request) -> bool:
    return _verify_token(request.cookies.get(COOKIE_NAME))


def _should_secure_cookie(request: web.Request) -> bool:
    if COOKIE_SECURE in ("1", "true", "yes", "on"):
        return True
    if COOKIE_SECURE in ("0", "false", "no", "off"):
        return False
    xf = request.headers.get("X-Forwarded-Proto", "").lower()
    return xf == "https" or request.scheme == "https"


def _set_session_cookie(resp: web.Response, request: web.Request) -> None:
    resp.set_cookie(
        COOKIE_NAME,
        _make_token(),
        max_age=SESSION_TTL,
        httponly=True,
        samesite="Lax",
        secure=_should_secure_cookie(request),
        path="/",
    )


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

@web.middleware
async def auth_middleware(request: web.Request, handler):
    path = request.path
    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return await handler(request)
    if _is_trusted_ip(request) or _is_authenticated(request):
        return await handler(request)
    if path.startswith("/api/"):
        raise web.HTTPUnauthorized(text="Unauthorized")
    raise web.HTTPFound("/login.html")


# ---------------------------------------------------------------------------
# Copilot runner
# ---------------------------------------------------------------------------

def _parse_copilot_jsonl(line: bytes) -> dict | None:
    """
    Parse one JSONL line from `copilot --output-format json`.

    Returns a structured event dict:
      {"kind": "chunk",     "text": str}          — streaming delta text
      {"kind": "tool_call", "tool": str}           — tool invocation
      {"kind": "result",    "session_id": str,
                            "tokens_out": int,
                            "premium_reqs": int}   — final result with session info
    Returns None for uninteresting events.
    """
    raw = line.decode("utf-8", errors="replace").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None

    t = obj.get("type", "")

    # Streaming text delta — the primary event type
    if t == "assistant.message_delta":
        text = obj.get("data", {}).get("deltaContent", "")
        return {"kind": "chunk", "text": text} if text else None

    # assistant.message is the *complete* final content — skip it since we
    # already received all the text via message_delta streaming events.

    # Tool invocations
    if t in ("tool_call", "assistant.tool_call", "tool.invocation", "tool.result"):
        data = obj.get("data", {})
        tool = data.get("name") or data.get("toolName") or data.get("tool") or t
        return {"kind": "tool_call", "tool": str(tool)}

    # Final result — carries the persistent session UUID
    if t == "result":
        usage = obj.get("usage", {})
        return {
            "kind": "result",
            "session_id": obj.get("sessionId", ""),
            "tokens_out": usage.get("outputTokens", 0),
            "premium_reqs": usage.get("premiumRequests", 0),
        }

    return None


async def _run_copilot_stream(
    prompt: str,
    copilot_session_id: str | None,
) -> AsyncIterator[dict]:
    """
    Yield structured event dicts from `copilot -p <prompt> --output-format json`.

    Uses --resume=<uuid> to continue an existing copilot session.
    The `result` event contains the session UUID for the next call.
    """
    cmd = [
        COPILOT_BIN,
        "-p", prompt,
        "--output-format", "json",
        "--silent",
        "--allow-all-tools",
    ]
    if copilot_session_id:
        cmd.append(f"--resume={copilot_session_id}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(WORKSPACE),
    )

    assert proc.stdout is not None
    try:
        async for line in proc.stdout:
            event = _parse_copilot_jsonl(line)
            if event:
                yield event
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------

async def handle_login(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON")
    if body.get("password") != PASSWORD:
        raise web.HTTPUnauthorized(text="Wrong password")
    resp = web.Response(text="ok")
    _set_session_cookie(resp, request)
    return resp


async def handle_auth_status(request: web.Request) -> web.Response:
    return web.json_response({"authenticated": _is_authenticated(request)})


async def handle_logout(request: web.Request) -> web.Response:
    resp = web.Response(text="ok")
    resp.del_cookie(COOKIE_NAME, path="/")
    return resp


async def handle_chat(request: web.Request) -> web.StreamResponse:
    """SSE endpoint: POST /api/chat  body: {"prompt": "...", "session_id": "..."}"""
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise web.HTTPBadRequest(text="prompt is required")

    browser_sid = (body.get("session_id") or "").strip() or str(uuid.uuid4())
    copilot_session = _copilot_sessions.get(browser_sid)

    # Store user message in history
    _chat_histories.setdefault(browser_sid, []).append({"role": "user", "content": prompt})

    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)

    # Send session id to client immediately
    await resp.write(f"data: {json.dumps({'type': 'session_id', 'session_id': browser_sid})}\n\n".encode())

    full_response = []
    try:
        async for event in _run_copilot_stream(prompt, copilot_session):
            kind = event["kind"]

            if kind == "chunk":
                full_response.append(event["text"])
                payload = json.dumps({"type": "chunk", "text": event["text"]})
                await resp.write(f"data: {payload}\n\n".encode())

            elif kind == "tool_call":
                payload = json.dumps({"type": "tool_call", "tool": event["tool"]})
                await resp.write(f"data: {payload}\n\n".encode())

            elif kind == "result":
                # Store the real copilot session UUID for next resume
                if event["session_id"]:
                    _copilot_sessions[browser_sid] = event["session_id"]
                payload = json.dumps({
                    "type": "usage",
                    "tokens_out": event["tokens_out"],
                    "premium_reqs": event["premium_reqs"],
                })
                await resp.write(f"data: {payload}\n\n".encode())

        _chat_histories[browser_sid].append({"role": "assistant", "content": "".join(full_response)})
        await resp.write(f"data: {json.dumps({'type': 'done'})}\n\n".encode())

    except Exception as exc:
        traceback.print_exc()
        err = json.dumps({"type": "error", "message": str(exc)})
        await resp.write(f"data: {err}\n\n".encode())

    await resp.write_eof()
    return resp


async def handle_history(request: web.Request) -> web.Response:
    """Return chat history for a browser session."""
    sid = request.rel_url.query.get("session_id", "")
    return web.json_response(_chat_histories.get(sid, []))


async def handle_sessions(request: web.Request) -> web.Response:
    """Return all known browser session ids with message counts."""
    result = [
        {"session_id": sid, "messages": len(msgs)}
        for sid, msgs in _chat_histories.items()
    ]
    return web.json_response(result)


async def handle_new_session(request: web.Request) -> web.Response:
    sid = str(uuid.uuid4())
    _chat_histories[sid] = []
    return web.json_response({"session_id": sid})


async def handle_delete_session(request: web.Request) -> web.Response:
    sid = request.match_info.get("session_id", "")
    _chat_histories.pop(sid, None)
    _copilot_sessions.pop(sid, None)
    return web.json_response({"ok": True})


async def handle_healthz(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def handle_index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(HTML_DIR / "index.html")


async def handle_login_page(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(HTML_DIR / "login.html")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app() -> web.Application:
    app = web.Application(middlewares=[auth_middleware])

    app.router.add_post("/api/login", handle_login)
    app.router.add_post("/api/logout", handle_logout)
    app.router.add_get("/api/auth/status", handle_auth_status)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_get("/api/history", handle_history)
    app.router.add_get("/api/sessions", handle_sessions)
    app.router.add_post("/api/sessions/new", handle_new_session)
    app.router.add_delete("/api/sessions/{session_id}", handle_delete_session)
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/", handle_index)
    app.router.add_get("/login", handle_login_page)

    app.router.add_static("/", HTML_DIR, name="static", show_index=False)
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if PASSWORD_AUTO:
        print(f"\n  ⚡ Auto-generated password: {PASSWORD}\n", flush=True)
    else:
        print(f"\n  🔒 Password auth enabled\n", flush=True)

    if TRUSTED_NETWORKS:
        nets = ", ".join(str(n) for n in TRUSTED_NETWORKS)
        print(f"  🌐 Trusted networks (no auth): {nets}", flush=True)

    # Verify copilot binary is accessible
    try:
        result = subprocess.run([COPILOT_BIN, "--version"], capture_output=True, text=True, timeout=5)
        print(f"  ✓ Copilot: {result.stdout.strip() or result.stderr.strip()}", flush=True)
    except FileNotFoundError:
        print(f"  ✗ Copilot binary not found at: {COPILOT_BIN}", file=sys.stderr)
        print("    Set COPILOT_BIN env var or mount the binary.", file=sys.stderr)
        sys.exit(1)

    print(f"  → http://{HOST}:{PORT}\n", flush=True)

    app = build_app()
    web.run_app(app, host=HOST, port=PORT, access_log=None)


if __name__ == "__main__":
    main()
