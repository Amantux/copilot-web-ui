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

# Map browser session_id -> copilot session UUID (for --resume=<uuid>)
_copilot_sessions: dict[str, str] = {}

# Per-session configuration
_session_configs: dict[str, dict] = {}

DEFAULT_SESSION_CONFIG: dict = {
    "workdir": str(WORKSPACE),
    "mode": "autopilot",          # interactive | plan | autopilot
    "yolo": True,                 # --yolo = allow-all-tools + allow-all-paths + allow-all-urls
    "model": "",                  # empty = use copilot default (claude-sonnet-4.6)
    "github_mcp_all": False,      # --enable-all-github-mcp-tools
    "reasoning_effort": "",       # low | medium | high | xhigh
    "max_continues": 0,           # 0 = unlimited (autopilot)
    "no_ask_user": False,         # --no-ask-user (fully autonomous)
    "label": "New chat",
}


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
      {"kind": "chunk",      "text": str}
      {"kind": "tool_call",  "tool": str, "args": dict}
      {"kind": "tool_done",  "tool": str, "success": bool}
      {"kind": "result",     "session_id": str, "tokens_out": int,
                             "premium_reqs": int, "files_modified": int,
                             "lines_added": int, "lines_removed": int}
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
    data = obj.get("data", {})

    # Streaming text delta — primary content event
    if t == "assistant.message_delta":
        text = data.get("deltaContent", "")
        return {"kind": "chunk", "text": text} if text else None

    # Tool execution starting — show badge immediately
    if t == "tool.execution_start":
        tool = data.get("toolName", data.get("name", "tool"))
        args = data.get("arguments", {})
        return {"kind": "tool_call", "tool": str(tool), "args": args}

    # Tool execution finished
    if t == "tool.execution_complete":
        tool = data.get("toolName", data.get("name", "tool"))
        success = data.get("success", True)
        return {"kind": "tool_done", "tool": str(tool), "success": success}

    # Final result — carries persistent session UUID and usage stats
    if t == "result":
        usage = obj.get("usage", {})
        changes = usage.get("codeChanges", {})
        return {
            "kind": "result",
            "session_id": obj.get("sessionId", ""),
            "tokens_out": usage.get("outputTokens", 0),
            "premium_reqs": usage.get("premiumRequests", 0),
            "files_modified": len(changes.get("filesModified", [])),
            "lines_added": changes.get("linesAdded", 0),
            "lines_removed": changes.get("linesRemoved", 0),
        }

    # assistant.message with toolRequests is superseded by tool.execution_start events
    # assistant.message content is covered by message_delta — skip both to avoid duplication
    return None


async def _run_copilot_stream(
    prompt: str,
    copilot_session_id: str | None,
    config: dict,
) -> AsyncIterator[dict]:
    """
    Yield structured event dicts from `copilot -p <prompt> --output-format json`.

    Builds the command from the per-session config (mode, yolo, model, workdir, etc.).
    Uses --resume=<uuid> to continue an existing copilot session.
    """
    cfg = {**DEFAULT_SESSION_CONFIG, **config}
    workdir = cfg["workdir"]

    # Validate / resolve workdir
    try:
        cwd = Path(workdir).resolve()
        if not cwd.exists():
            cwd = WORKSPACE
    except Exception:
        cwd = WORKSPACE

    cmd = [
        COPILOT_BIN,
        "-p", prompt,
        "--output-format", "json",
        "--silent",
    ]

    # Permissions
    if cfg["yolo"]:
        cmd.append("--yolo")
    else:
        # Minimum needed for non-interactive: allow tools + the workdir
        cmd.append("--allow-all-tools")
        if str(cwd) != str(WORKSPACE):
            cmd += ["--add-dir", str(cwd)]

    # Agent mode
    if cfg["mode"] in ("autopilot", "plan"):
        cmd += ["--mode", cfg["mode"]]

    if cfg["no_ask_user"]:
        cmd.append("--no-ask-user")

    # Model
    if cfg["model"]:
        cmd += ["--model", cfg["model"]]

    # GitHub MCP tools
    if cfg["github_mcp_all"]:
        cmd.append("--enable-all-github-mcp-tools")

    # Reasoning effort
    if cfg["reasoning_effort"]:
        cmd += ["--reasoning-effort", cfg["reasoning_effort"]]

    # Autopilot continue limit
    if cfg["max_continues"] and int(cfg["max_continues"]) > 0:
        cmd += ["--max-autopilot-continues", str(cfg["max_continues"])]

    # Session resume
    if copilot_session_id:
        cmd.append(f"--resume={copilot_session_id}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
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
    config = _session_configs.get(browser_sid, {})

    _chat_histories.setdefault(browser_sid, []).append({"role": "user", "content": prompt})

    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)

    await resp.write(f"data: {json.dumps({'type': 'session_id', 'session_id': browser_sid})}\n\n".encode())

    full_response = []
    try:
        async for event in _run_copilot_stream(prompt, copilot_session, config):
            kind = event["kind"]

            if kind == "chunk":
                full_response.append(event["text"])
                payload = json.dumps({"type": "chunk", "text": event["text"]})
                await resp.write(f"data: {payload}\n\n".encode())

            elif kind == "tool_call":
                payload = json.dumps({"type": "tool_call", "tool": event["tool"], "args": event.get("args", {})})
                await resp.write(f"data: {payload}\n\n".encode())

            elif kind == "tool_done":
                payload = json.dumps({"type": "tool_done", "tool": event["tool"], "success": event["success"]})
                await resp.write(f"data: {payload}\n\n".encode())

            elif kind == "result":
                if event["session_id"]:
                    _copilot_sessions[browser_sid] = event["session_id"]
                payload = json.dumps({
                    "type": "usage",
                    "tokens_out": event["tokens_out"],
                    "premium_reqs": event["premium_reqs"],
                    "files_modified": event["files_modified"],
                    "lines_added": event["lines_added"],
                    "lines_removed": event["lines_removed"],
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
    sid = request.rel_url.query.get("session_id", "")
    return web.json_response(_chat_histories.get(sid, []))


async def handle_sessions(request: web.Request) -> web.Response:
    result = []
    for sid, msgs in _chat_histories.items():
        cfg = _session_configs.get(sid, {})
        result.append({
            "session_id": sid,
            "messages": len(msgs),
            "label": cfg.get("label", "Chat"),
            "mode": cfg.get("mode", DEFAULT_SESSION_CONFIG["mode"]),
            "workdir": cfg.get("workdir", str(WORKSPACE)),
            "model": cfg.get("model", ""),
        })
    return web.json_response(result)


async def handle_new_session(request: web.Request) -> web.Response:
    sid = str(uuid.uuid4())
    _chat_histories[sid] = []
    # Accept config from request body
    try:
        body = await request.json()
    except Exception:
        body = {}

    cfg = {**DEFAULT_SESSION_CONFIG}
    for key in ("workdir", "mode", "yolo", "model", "github_mcp_all",
                "reasoning_effort", "max_continues", "no_ask_user", "label"):
        if key in body:
            cfg[key] = body[key]

    # Validate workdir
    try:
        wd = Path(cfg["workdir"]).resolve()
        cfg["workdir"] = str(wd) if wd.exists() else str(WORKSPACE)
    except Exception:
        cfg["workdir"] = str(WORKSPACE)

    _session_configs[sid] = cfg
    return web.json_response({"session_id": sid, "config": cfg})


async def handle_session_config(request: web.Request) -> web.Response:
    sid = request.match_info.get("session_id", "")
    if sid not in _chat_histories:
        raise web.HTTPNotFound(text="Session not found")
    return web.json_response(_session_configs.get(sid, DEFAULT_SESSION_CONFIG))


async def handle_delete_session(request: web.Request) -> web.Response:
    sid = request.match_info.get("session_id", "")
    _chat_histories.pop(sid, None)
    _copilot_sessions.pop(sid, None)
    _session_configs.pop(sid, None)
    return web.json_response({"ok": True})


async def handle_browse(request: web.Request) -> web.Response:
    """List directories at a given path for the directory picker."""
    path = request.rel_url.query.get("path", str(WORKSPACE))
    try:
        p = Path(path).resolve()
        if not p.exists() or not p.is_dir():
            p = WORKSPACE
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        dirs = []
        for e in entries:
            if e.is_dir() and not e.name.startswith("."):
                dirs.append({"name": e.name, "path": str(e)})
        return web.json_response({
            "path": str(p),
            "parent": str(p.parent) if p != p.parent else None,
            "dirs": dirs,
        })
    except PermissionError:
        raise web.HTTPForbidden(text="Permission denied")


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
    app.router.add_get("/api/sessions/{session_id}/config", handle_session_config)
    app.router.add_delete("/api/sessions/{session_id}", handle_delete_session)
    app.router.add_get("/api/browse", handle_browse)
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
