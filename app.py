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
    # Identity
    "label": "New chat",
    "session_name": "",           # --name (copilot session name)
    # Working directory
    "workdir": str(WORKSPACE),
    "add_dirs": [],               # --add-dir (extra allowed dirs)
    # Agent mode
    "mode": "autopilot",          # interactive | plan | autopilot
    # Permissions — granular (yolo sets all three)
    "yolo": True,                 # --yolo / --allow-all
    "allow_all_tools": False,     # --allow-all-tools (separate from yolo)
    "allow_all_paths": False,     # --allow-all-paths
    "allow_all_urls": False,      # --allow-all-urls
    "disallow_temp_dir": False,   # --disallow-temp-dir
    "allow_tool": [],             # --allow-tool patterns e.g. ["shell(git:*)", "write"]
    "deny_tool": [],              # --deny-tool patterns
    "allow_url": [],              # --allow-url domains
    "deny_url": [],               # --deny-url domains
    # Model & reasoning
    "model": "",                  # empty = copilot default
    "reasoning_effort": "",       # low | medium | high | xhigh
    # GitHub MCP
    "github_mcp_all": False,      # --enable-all-github-mcp-tools
    "add_github_mcp_tools": [],   # --add-github-mcp-tool patterns
    "add_github_mcp_toolsets": [],# --add-github-mcp-toolset names
    # Autopilot
    "max_continues": 0,           # 0 = unlimited
    "no_ask_user": False,         # --no-ask-user
    # Extra
    "agent": "",                  # --agent custom agent name
    "experimental": False,        # --experimental
    "no_custom_instructions": False,  # --no-custom-instructions
    "mcp_config": "",             # --additional-mcp-config JSON string
    # Tool visibility (--available-tools / --excluded-tools)
    "available_tools": [],        # --available-tools (which tools model can see)
    "excluded_tools": [],         # --excluded-tools (which tools to hide from model)
    # MCP control
    "disable_builtin_mcps": False, # --disable-builtin-mcps
    "disable_mcp_server": [],     # --disable-mcp-server (repeatable)
    # Reasoning
    "enable_reasoning_summaries": False, # --enable-reasoning-summaries
    # Security
    "secret_env_vars": [],        # --secret-env-vars
    # Plugins
    "plugin_dir": [],             # --plugin-dir (repeatable)
}

AVAILABLE_MODELS = [
    "claude-sonnet-4.6",
    "claude-sonnet-4.5",
    "claude-haiku-4.5",
    "claude-opus-4.7",
    "claude-opus-4.6",
    "claude-opus-4.5",
    "claude-sonnet-4",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "gpt-4.1",
]


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

    # ── Permissions ──────────────────────────────────────────────────────────
    if cfg.get("yolo"):
        cmd.append("--yolo")
    else:
        # Granular permission flags
        if cfg.get("allow_all_tools"):
            cmd.append("--allow-all-tools")
        else:
            # Non-interactive always needs at least allow-all-tools
            cmd.append("--allow-all-tools")

        if cfg.get("allow_all_paths"):
            cmd.append("--allow-all-paths")
        if cfg.get("allow_all_urls"):
            cmd.append("--allow-all-urls")
        if cfg.get("disallow_temp_dir"):
            cmd.append("--disallow-temp-dir")

        # Additional directories
        if str(cwd) != str(WORKSPACE):
            cmd += ["--add-dir", str(cwd)]
        for extra_dir in cfg.get("add_dirs") or []:
            d = extra_dir.strip()
            if d and d != str(cwd):
                cmd += ["--add-dir", d]

        # Tool allow/deny patterns (e.g. "shell(git:*)", "write")
        for pat in cfg.get("allow_tool") or []:
            pat = pat.strip()
            if pat:
                cmd += ["--allow-tool", pat]
        for pat in cfg.get("deny_tool") or []:
            pat = pat.strip()
            if pat:
                cmd += ["--deny-tool", pat]

        # URL allow/deny lists
        for url in cfg.get("allow_url") or []:
            url = url.strip()
            if url:
                cmd += ["--allow-url", url]
        for url in cfg.get("deny_url") or []:
            url = url.strip()
            if url:
                cmd += ["--deny-url", url]

    # ── Agent mode ───────────────────────────────────────────────────────────
    if cfg.get("mode") in ("autopilot", "plan", "interactive"):
        cmd += ["--mode", cfg["mode"]]

    if cfg.get("no_ask_user"):
        cmd.append("--no-ask-user")

    # ── Model & reasoning ────────────────────────────────────────────────────
    if cfg.get("model"):
        cmd += ["--model", cfg["model"]]
    if cfg.get("reasoning_effort"):
        cmd += ["--reasoning-effort", cfg["reasoning_effort"]]

    # ── GitHub MCP tools ─────────────────────────────────────────────────────
    if cfg.get("github_mcp_all"):
        cmd.append("--enable-all-github-mcp-tools")
    else:
        for tool in cfg.get("add_github_mcp_tools") or []:
            t = tool.strip()
            if t:
                cmd += ["--add-github-mcp-tool", t]
        for toolset in cfg.get("add_github_mcp_toolsets") or []:
            ts = toolset.strip()
            if ts:
                cmd += ["--add-github-mcp-toolset", ts]

    # ── Autopilot ────────────────────────────────────────────────────────────
    if cfg.get("max_continues") and int(cfg["max_continues"]) > 0:
        cmd += ["--max-autopilot-continues", str(cfg["max_continues"])]

    # ── Custom agent ─────────────────────────────────────────────────────────
    if cfg.get("agent"):
        cmd += ["--agent", cfg["agent"]]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if cfg.get("experimental"):
        cmd.append("--experimental")
    if cfg.get("no_custom_instructions"):
        cmd.append("--no-custom-instructions")
    if cfg.get("mcp_config", "").strip():
        cmd += ["--additional-mcp-config", cfg["mcp_config"].strip()]

    # ── Session resume ───────────────────────────────────────────────────────
    if copilot_session_id:
        cmd.append(f"--resume={copilot_session_id}")

    # ── Tool visibility ───────────────────────────────────────────────────────
    for tool in cfg.get("available_tools") or []:
        t = tool.strip()
        if t:
            cmd += ["--available-tools", t]
    for tool in cfg.get("excluded_tools") or []:
        t = tool.strip()
        if t:
            cmd += ["--excluded-tools", t]

    # ── MCP control ───────────────────────────────────────────────────────────
    if cfg.get("disable_builtin_mcps"):
        cmd.append("--disable-builtin-mcps")
    for srv in cfg.get("disable_mcp_server") or []:
        s = srv.strip()
        if s:
            cmd += ["--disable-mcp-server", s]

    # ── Reasoning summaries ───────────────────────────────────────────────────
    if cfg.get("enable_reasoning_summaries"):
        cmd.append("--enable-reasoning-summaries")

    # ── Secret env vars ───────────────────────────────────────────────────────
    if cfg.get("secret_env_vars"):
        cmd += ["--secret-env-vars", ",".join(v.strip() for v in cfg["secret_env_vars"] if v.strip())]

    # ── Plugin directories ────────────────────────────────────────────────────
    for pd in cfg.get("plugin_dir") or []:
        p = pd.strip()
        if p:
            cmd += ["--plugin-dir", p]

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


async def handle_version(_request: web.Request) -> web.Response:
    proc = await asyncio.create_subprocess_exec(
        COPILOT_BIN, "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    version_str = stdout.decode().strip()
    return web.json_response({"version": version_str})


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
    try:
        body = await request.json()
    except Exception:
        body = {}

    cfg = {**DEFAULT_SESSION_CONFIG}
    for key in DEFAULT_SESSION_CONFIG:
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
    return web.json_response(_session_configs.get(sid, {**DEFAULT_SESSION_CONFIG}))


async def handle_update_session_config(request: web.Request) -> web.Response:
    """PATCH /api/sessions/:id/config — update config fields mid-session."""
    sid = request.match_info.get("session_id", "")
    if sid not in _chat_histories:
        raise web.HTTPNotFound(text="Session not found")
    try:
        body = await request.json()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON")

    cfg = _session_configs.get(sid, {**DEFAULT_SESSION_CONFIG})
    allowed = set(DEFAULT_SESSION_CONFIG.keys())
    for key, val in body.items():
        if key in allowed:
            cfg[key] = val
    _session_configs[sid] = cfg
    return web.json_response(cfg)


async def handle_rename_session(request: web.Request) -> web.Response:
    """PUT /api/sessions/:id/rename — rename a session."""
    sid = request.match_info.get("session_id", "")
    if sid not in _chat_histories:
        raise web.HTTPNotFound(text="Session not found")
    try:
        body = await request.json()
        name = str(body.get("name", "")).strip()
    except Exception:
        raise web.HTTPBadRequest(text="Invalid JSON")
    if not name:
        raise web.HTTPBadRequest(text="Name required")
    cfg = _session_configs.setdefault(sid, {**DEFAULT_SESSION_CONFIG})
    cfg["label"] = name
    return web.json_response({"ok": True, "label": name})


async def handle_undo_session(request: web.Request) -> web.Response:
    """DELETE /api/sessions/:id/last — remove the last user+assistant exchange."""
    sid = request.match_info.get("session_id", "")
    if sid not in _chat_histories:
        raise web.HTTPNotFound(text="Session not found")
    msgs = _chat_histories[sid]
    removed = 0
    # Remove last assistant message
    if msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
        removed += 1
    # Remove last user message
    if msgs and msgs[-1]["role"] == "user":
        msgs.pop()
        removed += 1
    # If we undo, also clear the copilot session so next message starts fresh from history
    if removed > 0:
        _copilot_sessions.pop(sid, None)
    return web.json_response({"ok": True, "removed": removed, "remaining": len(msgs)})


async def handle_export_session(request: web.Request) -> web.Response:
    """GET /api/sessions/:id/export — export conversation as markdown."""
    sid = request.match_info.get("session_id", "")
    if sid not in _chat_histories:
        raise web.HTTPNotFound(text="Session not found")
    cfg = _session_configs.get(sid, {})
    msgs = _chat_histories[sid]
    label = cfg.get("label", "Copilot Session")
    lines = [
        f"# {label}",
        "",
        f"> Session ID: `{sid}`  ",
        f"> Mode: `{cfg.get('mode','autopilot')}`  ",
        f"> Model: `{cfg.get('model','claude-sonnet-4.6')}`  ",
        f"> Working directory: `{cfg.get('workdir','')}`",
        "",
        "---",
        "",
    ]
    for msg in msgs:
        role = "**You**" if msg["role"] == "user" else "**GitHub Copilot**"
        lines.append(f"### {role}\n")
        lines.append(msg["content"])
        lines.append("")
        lines.append("---")
        lines.append("")
    md = "\n".join(lines)
    filename = f"copilot-session-{sid[:8]}.md"
    return web.Response(
        body=md.encode("utf-8"),
        content_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def handle_models(_request: web.Request) -> web.Response:
    """GET /api/models — list available models."""
    return web.json_response(AVAILABLE_MODELS)


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
    app.router.add_get("/api/version", handle_version)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_get("/api/history", handle_history)
    app.router.add_get("/api/sessions", handle_sessions)
    app.router.add_post("/api/sessions/new", handle_new_session)
    app.router.add_get("/api/sessions/{session_id}/config", handle_session_config)
    app.router.add_patch("/api/sessions/{session_id}/config", handle_update_session_config)
    app.router.add_put("/api/sessions/{session_id}/rename", handle_rename_session)
    app.router.add_delete("/api/sessions/{session_id}/last", handle_undo_session)
    app.router.add_get("/api/sessions/{session_id}/export", handle_export_session)
    app.router.add_delete("/api/sessions/{session_id}", handle_delete_session)
    app.router.add_get("/api/models", handle_models)
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
