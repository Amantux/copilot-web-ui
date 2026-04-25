"""Tests for copilot-web-ui (app.py).

Covers:
  - JSONL text extraction (_extract_text_from_jsonl)
  - Auth token helpers (_sign, _make_token, _verify_token)
  - HTTP endpoints via aiohttp test client (login, logout, auth status,
    healthz, sessions CRUD, chat SSE with mocked copilot subprocess)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient

# ---------------------------------------------------------------------------
# Load app module with controlled env so module-level globals are predictable
# ---------------------------------------------------------------------------
os.environ.setdefault("COPILOT_WEB_PASSWORD", "testpass")
os.environ.setdefault("COPILOT_WEB_SECRET", "a" * 64)
os.environ.setdefault("COPILOT_BIN", "copilot")
os.environ.setdefault("COPILOT_WORKSPACE", "/tmp/cwui-test-workspace")

import app as appmod  # noqa: E402  (must be after env setup)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def make_client(aiohttp_client, clear_state: bool = True):
    if clear_state:
        appmod._chat_histories.clear()
        appmod._copilot_sessions.clear()
    return await aiohttp_client(appmod.build_app())


def _auth_header(client: TestClient) -> dict:
    token = appmod._make_token()
    return {"Cookie": f"{appmod.COOKIE_NAME}={token}"}


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE body into a list of data-payload dicts."""
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# Fake subprocess factory
# ---------------------------------------------------------------------------

FAKE_TEXT_CHUNK = json.dumps({
    "type": "content_block_delta",
    "delta": {"type": "text_delta", "text": "Hello from Copilot!"},
})
FAKE_STOP = json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})


def _make_mock_process(output_lines: list[str]):
    """Return a mock that behaves like asyncio.create_subprocess_exec output."""

    class _FakeStream:
        def __init__(self, lines):
            self._iter = iter(lines)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._iter).encode()
            except StopIteration:
                raise StopAsyncIteration

    proc = MagicMock()
    proc.stdout = _FakeStream(output_lines)
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    return proc


# ===========================================================================
# 1. JSONL text extraction
# ===========================================================================

class TestExtractTextFromJsonl:
    def _call(self, data: str) -> str:
        return appmod._extract_text_from_jsonl(data.encode())

    def test_empty_line_returns_empty(self):
        assert self._call("") == ""
        assert self._call("   ") == ""

    def test_content_block_delta_text(self):
        line = json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello world"},
        })
        assert self._call(line) == "Hello world"

    def test_content_block_delta_non_text_returns_empty(self):
        line = json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        })
        assert self._call(line) == ""

    def test_message_type_list_content(self):
        line = json.dumps({
            "type": "message",
            "content": [
                {"type": "text", "text": "Part one "},
                {"type": "text", "text": "part two"},
            ],
        })
        assert self._call(line) == "Part one part two"

    def test_message_type_string_content(self):
        line = json.dumps({"type": "message", "content": "Simple string"})
        assert self._call(line) == "Simple string"

    def test_role_assistant_style(self):
        line = json.dumps({"role": "assistant", "content": "Direct content"})
        assert self._call(line) == "Direct content"

    def test_openai_delta_content_field(self):
        line = json.dumps({
            "choices": [{"delta": {"content": "OpenAI chunk"}, "finish_reason": None}]
        })
        assert self._call(line) == "OpenAI chunk"

    def test_openai_delta_text_field(self):
        line = json.dumps({
            "choices": [{"delta": {"text": "Alt field"}}]
        })
        assert self._call(line) == "Alt field"

    def test_unknown_json_shape_returns_empty(self):
        line = json.dumps({"type": "message_start", "message": {}})
        assert self._call(line) == ""

    def test_plain_text_fallback(self):
        # Non-JSON, non-brace lines pass through as plain text
        assert self._call("plain text line") == "plain text line"

    def test_malformed_json_brace_suppressed(self):
        # Malformed JSON that starts with { is suppressed (not forwarded raw)
        assert self._call("{broken json") == ""

    def test_skips_non_text_content_blocks_in_list(self):
        line = json.dumps({
            "type": "message",
            "content": [
                {"type": "tool_use", "id": "tu_1"},
                {"type": "text", "text": "After tool"},
            ],
        })
        assert self._call(line) == "After tool"


# ===========================================================================
# 2. Auth token helpers
# ===========================================================================

class TestAuthHelpers:
    def test_sign_is_deterministic(self):
        assert appmod._sign("payload") == appmod._sign("payload")

    def test_sign_differs_for_different_payloads(self):
        assert appmod._sign("aaa") != appmod._sign("bbb")

    def test_make_and_verify_token_roundtrip(self):
        token = appmod._make_token()
        assert appmod._verify_token(token) is True

    def test_verify_none_returns_false(self):
        assert appmod._verify_token(None) is False

    def test_verify_empty_string_returns_false(self):
        assert appmod._verify_token("") is False

    def test_verify_garbage_returns_false(self):
        assert appmod._verify_token("garbage.no.dots.enough") is False

    def test_verify_expired_token(self):
        expired = appmod._make_token(ttl=-10)
        assert appmod._verify_token(expired) is False

    def test_verify_tampered_signature(self):
        token = appmod._make_token()
        parts = token.split(".")
        parts[-1] = "AAAAAAAAAAAAAAAA"
        assert appmod._verify_token(".".join(parts)) is False

    def test_verify_tampered_expiry(self):
        token = appmod._make_token()
        version, _exp, sig = token.split(".")
        # Reset expiry to epoch 0 while keeping original signature
        assert appmod._verify_token(f"{version}.0.{sig}") is False

    def test_verify_wrong_version(self):
        token = appmod._make_token()
        _v, exp, sig = token.split(".")
        assert appmod._verify_token(f"v99.{exp}.{sig}") is False


# ===========================================================================
# 3. HTTP endpoints
# ===========================================================================

class TestHealthz:
    @pytest.mark.asyncio
    async def test_healthz_ok(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.get("/healthz")
        assert resp.status == 200
        assert await resp.text() == "ok"

    @pytest.mark.asyncio
    async def test_healthz_no_auth_required(self, aiohttp_client):
        """Healthz must be reachable without a session cookie."""
        client = await make_client(aiohttp_client)
        resp = await client.get("/healthz")
        assert resp.status == 200


class TestLoginEndpoint:
    @pytest.mark.asyncio
    async def test_correct_password_sets_cookie(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.post("/api/login", json={"password": "testpass"})
        assert resp.status == 200
        assert appmod.COOKIE_NAME in resp.cookies

    @pytest.mark.asyncio
    async def test_wrong_password_is_401(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.post("/api/login", json={"password": "wrong"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_bad_json_body_is_400(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.post(
            "/api/login", data="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400


class TestAuthStatus:
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_false(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.get("/api/auth/status")
        data = await resp.json()
        assert data["authenticated"] is False

    @pytest.mark.asyncio
    async def test_authenticated_returns_true(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        token = appmod._make_token()
        resp = await client.get(
            "/api/auth/status",
            headers={"Cookie": f"{appmod.COOKIE_NAME}={token}"},
        )
        data = await resp.json()
        assert data["authenticated"] is True


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_protected_api_without_cookie_is_401(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.get("/api/sessions")
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_login_html_accessible_without_auth(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.get("/login.html")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_root_redirects_to_login_without_auth(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.get("/", allow_redirects=False)
        assert resp.status in (301, 302)
        assert "/login" in resp.headers.get("Location", "")


class TestSessionEndpoints:
    @pytest.mark.asyncio
    async def test_new_session_creates_entry(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        resp = await client.post("/api/sessions/new", headers=h)
        assert resp.status == 200
        data = await resp.json()
        assert "session_id" in data
        assert data["session_id"] in appmod._chat_histories

    @pytest.mark.asyncio
    async def test_list_sessions_includes_created(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r1 = await client.post("/api/sessions/new", headers=h)
        r2 = await client.post("/api/sessions/new", headers=h)
        s1 = (await r1.json())["session_id"]
        s2 = (await r2.json())["session_id"]

        resp = await client.get("/api/sessions", headers=h)
        ids = [s["session_id"] for s in await resp.json()]
        assert s1 in ids and s2 in ids

    @pytest.mark.asyncio
    async def test_delete_session_removes_entry(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        resp = await client.delete(f"/api/sessions/{sid}", headers=h)
        assert resp.status == 200
        assert sid not in appmod._chat_histories

    @pytest.mark.asyncio
    async def test_history_empty_for_new_session(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        resp = await client.get(f"/api/history?session_id={sid}", headers=h)
        assert resp.status == 200
        assert await resp.json() == []

    @pytest.mark.asyncio
    async def test_history_unknown_session_returns_empty(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        resp = await client.get("/api/history?session_id=ghost", headers=h)
        assert resp.status == 200
        assert await resp.json() == []


# ===========================================================================
# 4. Chat SSE endpoint (mocked copilot subprocess)
# ===========================================================================

class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_requires_auth(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.post("/api/chat", json={"prompt": "hi"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_chat_empty_prompt_is_400(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        resp = await client.post(
            "/api/chat", json={"prompt": "   "},
            headers=_auth_header(None),
        )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_chat_bad_json_is_400(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(None)
        h["Content-Type"] = "application/json"
        resp = await client.post("/api/chat", data="notjson", headers=h)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_chat_sse_content_type(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        fake_proc = _make_mock_process([FAKE_TEXT_CHUNK, FAKE_STOP])
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            resp = await client.post(
                "/api/chat",
                json={"prompt": "hello", "session_id": sid},
                headers=h,
            )
        assert resp.status == 200
        assert "text/event-stream" in resp.headers["Content-Type"]

    @pytest.mark.asyncio
    async def test_chat_sse_events_structure(self, aiohttp_client):
        """SSE body must contain session_id, chunk, and done events."""
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        fake_proc = _make_mock_process([FAKE_TEXT_CHUNK, FAKE_STOP])
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            resp = await client.post(
                "/api/chat",
                json={"prompt": "test", "session_id": sid},
                headers=h,
            )
        events = _parse_sse(await resp.text())
        types_ = {e.get("type") for e in events}
        assert "session_id" in types_
        assert "chunk" in types_
        assert "done" in types_

    @pytest.mark.asyncio
    async def test_chat_sse_chunk_text_content(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        fake_proc = _make_mock_process([FAKE_TEXT_CHUNK])
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            resp = await client.post(
                "/api/chat",
                json={"prompt": "hi", "session_id": sid},
                headers=h,
            )
        events = _parse_sse(await resp.text())
        chunks = [e["text"] for e in events if e.get("type") == "chunk"]
        assert "".join(chunks) == "Hello from Copilot!"

    @pytest.mark.asyncio
    async def test_chat_records_user_message_in_history(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        fake_proc = _make_mock_process([FAKE_TEXT_CHUNK])
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            await client.post(
                "/api/chat",
                json={"prompt": "my question", "session_id": sid},
                headers=h,
            )
        history = appmod._chat_histories[sid]
        user_msgs = [m for m in history if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "my question"

    @pytest.mark.asyncio
    async def test_chat_records_assistant_message_in_history(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        fake_proc = _make_mock_process([FAKE_TEXT_CHUNK, FAKE_STOP])
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            await client.post(
                "/api/chat",
                json={"prompt": "q", "session_id": sid},
                headers=h,
            )
        history = appmod._chat_histories[sid]
        asst_msgs = [m for m in history if m["role"] == "assistant"]
        assert len(asst_msgs) == 1
        assert asst_msgs[0]["content"] == "Hello from Copilot!"

    @pytest.mark.asyncio
    async def test_chat_invokes_copilot_with_correct_flags(self, aiohttp_client):
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        fake_proc = _make_mock_process([])
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc) as mock_exec:
            await client.post(
                "/api/chat",
                json={"prompt": "do something", "session_id": sid},
                headers=h,
            )
            args = mock_exec.call_args[0]
            assert args[0] == appmod.COPILOT_BIN
            assert "-p" in args
            assert "do something" in args
            assert "--output-format" in args
            assert "json" in args
            assert "--allow-all-tools" in args
            assert "--silent" in args

    @pytest.mark.asyncio
    async def test_chat_auto_creates_session_if_omitted(self, aiohttp_client):
        """Omitting session_id should still work and auto-create a session."""
        client = await make_client(aiohttp_client)
        h = _auth_header(client)

        fake_proc = _make_mock_process([FAKE_TEXT_CHUNK])
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            resp = await client.post(
                "/api/chat",
                json={"prompt": "auto"},
                headers=h,
            )
        events = _parse_sse(await resp.text())
        sid_events = [e for e in events if e.get("type") == "session_id"]
        assert len(sid_events) == 1
        assert sid_events[0]["session_id"] in appmod._chat_histories

    @pytest.mark.asyncio
    async def test_chat_stores_copilot_session_name(self, aiohttp_client):
        """Copilot session name should be persisted for --resume on next turn."""
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        fake_proc = _make_mock_process([FAKE_TEXT_CHUNK])
        with patch("asyncio.create_subprocess_exec", return_value=fake_proc):
            await client.post(
                "/api/chat",
                json={"prompt": "first message", "session_id": sid},
                headers=h,
            )
        # After first chat, a copilot session name should be stored
        assert sid in appmod._copilot_sessions
        assert appmod._copilot_sessions[sid].startswith("cwui-")

    @pytest.mark.asyncio
    async def test_chat_uses_resume_flag_on_second_message(self, aiohttp_client):
        """Second message in same session should pass --resume to copilot."""
        client = await make_client(aiohttp_client)
        h = _auth_header(client)
        r = await client.post("/api/sessions/new", headers=h)
        sid = (await r.json())["session_id"]

        fake_proc1 = _make_mock_process([FAKE_TEXT_CHUNK])
        fake_proc2 = _make_mock_process([FAKE_TEXT_CHUNK])

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc1):
            await client.post("/api/chat", json={"prompt": "msg1", "session_id": sid}, headers=h)

        copilot_sid = appmod._copilot_sessions[sid]

        with patch("asyncio.create_subprocess_exec", return_value=fake_proc2) as mock2:
            await client.post("/api/chat", json={"prompt": "msg2", "session_id": sid}, headers=h)
            args = mock2.call_args[0]
            assert "--resume" in args
            assert copilot_sid in args
