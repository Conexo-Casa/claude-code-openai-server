"""The concurrency ceiling must cover the autonomous (tool-less) path too.

Patch 1 gated only ConversationManager.create(), so tool-less completions could
spawn `claude` subprocesses without limit — request-scoped, but N concurrent
requests still meant N processes and the same OOM. These tests drive the real
HTTP surface with a stub session so nothing is actually spawned.
"""

import asyncio

import httpx
import pytest

from app import main as main_mod
from app.claude_session import STREAM_CLOSED
from app.config import Settings
from app.conversation import ConversationManager
from app.mcp_bridge import McpBridge


class _StubSession:
    """Stands in for ClaudeSession. Class attrs let subclasses vary behaviour."""

    delay = 0.0
    fail = False

    def __init__(self, **kw):
        self.closed = False

    async def start(self):
        if type(self).fail:
            raise RuntimeError("spawn failed")
        if type(self).delay:
            await asyncio.sleep(type(self).delay)

    async def send_user_turn(self, content):
        pass

    async def next_event(self, timeout=None):
        return STREAM_CLOSED

    async def aclose(self):
        self.closed = True

    @property
    def running(self):
        return not self.closed


def _settings(**over):
    base = {"host": "127.0.0.1", "api_key": None}
    base.update(over)
    return Settings(**base)


def _app(monkeypatch, cap, session_cls=_StubSession):
    import app.routes.chat as chatmod

    s = _settings(max_concurrent_conversations=cap)
    monkeypatch.setattr(main_mod, "get_settings", lambda: s)
    monkeypatch.setattr(chatmod, "get_settings", lambda: s)
    monkeypatch.setattr(chatmod, "ClaudeSession", session_cls)
    app = main_mod.create_app()
    # The lifespan normally builds this; ASGITransport does not run lifespan.
    app.state.conv_manager = ConversationManager(McpBridge(), s)
    return app


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _body(text="hi", **over):
    b = {"model": "claude-opus-5", "messages": [{"role": "user", "content": text}]}
    b.update(over)
    return b


# ── the ceiling now applies here ─────────────────────────────────────────────


async def test_autonomous_turn_is_capped(monkeypatch):
    """Two concurrent tool-less turns against a cap of 1: one wins, one 429s."""

    class Slow(_StubSession):
        delay = 0.05

    app = _app(monkeypatch, 1, Slow)
    async with _client(app) as c:
        r1, r2 = await asyncio.gather(
            c.post("/v1/chat/completions", json=_body("a")),
            c.post("/v1/chat/completions", json=_body("b")),
        )

    assert sorted([r1.status_code, r2.status_code]) == [200, 429]
    refused = r1 if r1.status_code == 429 else r2
    assert refused.json()["error"]["code"] == "max_concurrent_conversations"


async def test_autonomous_slot_returned_after_each_request(monkeypatch):
    """Sequential turns reuse the one slot: request-scoped means released."""
    app = _app(monkeypatch, 1)
    async with _client(app) as c:
        for i in range(3):
            r = await c.post("/v1/chat/completions", json=_body(f"t{i}"))
            assert r.status_code == 200, r.text
    assert app.state.conv_manager._reserved == 0


async def test_streaming_turn_releases_slot(monkeypatch):
    """The slot goes back when the SSE generator finishes, not when it starts."""
    app = _app(monkeypatch, 1)
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json=_body("s", stream=True))
        assert r.status_code == 200
        assert r.text  # drain, so the generator's finally runs
    assert app.state.conv_manager._reserved == 0


async def test_slot_returned_when_spawn_fails(monkeypatch):
    """A failed spawn must not leak a lane — that would shrink the cap forever."""

    class Boom(_StubSession):
        fail = True

    app = _app(monkeypatch, 1, Boom)
    async with _client(app) as c:
        with pytest.raises(RuntimeError):
            await c.post("/v1/chat/completions", json=_body("boom"))
    assert app.state.conv_manager._reserved == 0


# ── one ceiling, shared by both paths ────────────────────────────────────────


async def test_tool_and_autonomous_share_one_ceiling(monkeypatch):
    """A slot held by the tool path blocks an autonomous turn, and releasing frees it.

    This is the point of making the slot API public rather than giving the route
    its own counter: two independent caps of 12 would still allow 24 processes.
    """
    app = _app(monkeypatch, 1)
    mgr = app.state.conv_manager

    assert await mgr.acquire_slot() is True  # stand in for a live tool lane
    async with _client(app) as c:
        blocked = await c.post("/v1/chat/completions", json=_body("blocked"))
    assert blocked.status_code == 429

    await mgr.release_slot()
    async with _client(app) as c:
        ok = await c.post("/v1/chat/completions", json=_body("now fine"))
    assert ok.status_code == 200


# ── observability: the cap is readable, not just loggable when exceeded ──────


async def test_healthz_reports_lane_usage(monkeypatch):
    app = _app(monkeypatch, 12)
    async with _client(app) as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["conversations"] == {"live": 0, "spawning": 0, "cap": 12}


async def test_healthz_counts_a_held_slot(monkeypatch):
    app = _app(monkeypatch, 12)
    await app.state.conv_manager.acquire_slot()
    async with _client(app) as c:
        r = await c.get("/healthz")
    assert r.json()["conversations"]["spawning"] == 1


async def test_healthz_survives_missing_manager(monkeypatch):
    """A health check during startup must not 500."""
    app = _app(monkeypatch, 12)
    del app.state.conv_manager
    async with _client(app) as c:
        r = await c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "conversations" not in r.json()
