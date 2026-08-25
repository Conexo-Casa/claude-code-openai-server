"""Tests for the max-concurrent-conversations ceiling.

Every live conversation owns a `claude` subprocess holding ~300 MB, so this cap
is what stops a runaway client from OOM-killing the server instead of getting a
429. A stub session stands in for ClaudeSession so nothing is really spawned.
"""

import asyncio
from pathlib import Path

import pytest

from app.claude_session import STREAM_CLOSED
from app.conversation import RUNNING, SUSPENDED, ConversationManager
from app.errors import OpenAIError
from app.mcp_bridge import McpBridge
from app.openai_models import (
    ChatCompletionRequest,
    ChatMessage,
    FunctionCall,
    ToolCall,
)


class _StubSession:
    """Swallows ClaudeSession's kwargs and spawns nothing."""

    def __init__(self, **kw):
        self.closed = False
        self.sent_turns = []

    async def start(self):
        pass

    async def send_user_turn(self, content):
        self.sent_turns.append(content)

    async def next_event(self, timeout=None):
        return STREAM_CLOSED

    async def aclose(self):
        self.closed = True

    @property
    def running(self):
        return not self.closed


def make_settings(**over):
    from app.config import Settings

    base = dict(suspended_ttl_s=300, idle_session_ttl_s=900, gc_interval_s=30,
                request_timeout_s=30, permission_mode="bypassPermissions", port=8787)
    base.update(over)
    return Settings(**base)


def fresh_req(text="hi"):
    return ChatCompletionRequest(messages=[ChatMessage(role="user", content=text)])


def _patch_session(monkeypatch, cls=_StubSession):
    import app.conversation as convmod

    monkeypatch.setattr(convmod, "ClaudeSession", cls)


def _mgr(monkeypatch, **over):
    _patch_session(monkeypatch)
    return ConversationManager(McpBridge(), make_settings(**over))


async def _create(mgr, text="hi"):
    return await mgr.create(
        fresh_req(text), model="claude-opus-5", workdir=Path("/tmp"), effort=None
    )


# ── the ceiling ──────────────────────────────────────────────────────────────


async def test_cap_refuses_beyond_ceiling(monkeypatch):
    mgr = _mgr(monkeypatch, max_concurrent_conversations=2)
    await _create(mgr, "one")
    await _create(mgr, "two")

    with pytest.raises(OpenAIError) as ei:
        await _create(mgr, "three")

    err = ei.value
    assert err.status_code == 429
    assert err.type == "rate_limit_error"
    assert err.code == "max_concurrent_conversations"
    # The refused turn must not have registered or spawned anything.
    assert len(mgr._conversations) == 2
    assert mgr._reserved == 0


async def test_cap_zero_disables_the_ceiling(monkeypatch):
    mgr = _mgr(monkeypatch, max_concurrent_conversations=0)
    for i in range(5):
        await _create(mgr, f"t{i}")
    assert len(mgr._conversations) == 5


async def test_freed_slot_is_reusable(monkeypatch):
    """Closing a conversation returns its slot to the pool."""
    mgr = _mgr(monkeypatch, max_concurrent_conversations=1)
    conv = await _create(mgr, "one")
    with pytest.raises(OpenAIError):
        await _create(mgr, "two")

    await mgr._close(conv)
    assert len(mgr._conversations) == 0
    await _create(mgr, "three")  # the slot is free again
    assert len(mgr._conversations) == 1


# ── the reservation (why a bare len() check is not enough) ───────────────────


async def test_concurrent_stampede_respects_cap(monkeypatch):
    """cap+3 simultaneous fresh turns: exactly cap win, the rest get 429.

    Regression guard for the TOCTOU window. session.start() takes real time, so
    checking len(_conversations) without reserving would let every concurrent
    caller past the gate before the first one registers.
    """
    cap = 3

    class Slow(_StubSession):
        async def start(self):
            await asyncio.sleep(0.05)

    _patch_session(monkeypatch, Slow)
    mgr = ConversationManager(
        McpBridge(), make_settings(max_concurrent_conversations=cap)
    )

    results = await asyncio.gather(
        *(_create(mgr, f"t{i}") for i in range(cap + 3)), return_exceptions=True
    )
    refused = [r for r in results if isinstance(r, OpenAIError)]
    admitted = [r for r in results if not isinstance(r, Exception)]

    assert len(admitted) == cap
    assert len(refused) == 3
    assert all(r.status_code == 429 for r in refused)
    assert len(mgr._conversations) == cap
    assert mgr._reserved == 0


async def test_slot_released_when_spawn_fails(monkeypatch):
    """A failed spawn must not leak its reservation and wedge the server."""

    class Boom(_StubSession):
        async def start(self):
            raise RuntimeError("spawn failed")

    _patch_session(monkeypatch, Boom)
    mgr = ConversationManager(McpBridge(), make_settings(max_concurrent_conversations=1))

    with pytest.raises(RuntimeError):
        await _create(mgr, "boom")
    assert mgr._reserved == 0
    assert len(mgr._conversations) == 0

    # The single slot is still usable afterwards.
    _patch_session(monkeypatch, _StubSession)
    await _create(mgr, "after")
    assert len(mgr._conversations) == 1


# ── continuations are never refused ──────────────────────────────────────────


async def test_continuation_not_refused_at_cap(monkeypatch):
    """A continuation reattaches to a live subprocess, so the cap must ignore it.

    Refusing one would strand a suspended turn whose tool results have nowhere
    to go — the process stays parked until the TTL reaps it.
    """
    mgr = _mgr(monkeypatch, max_concurrent_conversations=1)
    conv = await _create(mgr, "first")  # exactly at cap now

    # Park it as a real suspended turn would: state SUSPENDED, one pending id.
    conv.state = SUSPENDED
    mgr._pending_index["call_1"] = conv.conv_id

    cont = ChatCompletionRequest(messages=[
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content=None, tool_calls=[
            ToolCall(id="call_1", function=FunctionCall(name="f", arguments="{}"))]),
        ChatMessage(role="tool", tool_call_id="call_1", content="result"),
    ])
    assert ConversationManager.is_continuation(cont) is True

    resumed = await mgr.resume(cont)  # must not raise 429
    assert resumed.conv_id == conv.conv_id
    assert resumed.state == RUNNING
