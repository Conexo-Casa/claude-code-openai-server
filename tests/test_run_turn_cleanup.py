"""`run_turn` must finish cleaning up before it yields its terminal chunk.

Every terminal branch in `run_turn` is shaped:

    yield DoneChunk(...)        # or ErrorChunk
    await self._close(conv)
    return

That only works if the consumer resumes the generator after the terminal chunk.
Neither consumer in `routes/chat.py` does — `_tool_stream` `break`s on
DoneChunk/ErrorChunk/ToolCallsChunk, and `_tool_collect` `return`s or `raise`s.
So the `_close(conv)` line never runs at that moment; it runs only when the
abandoned async generator is finalized, which is GC-scheduled and arbitrarily
later.

What `_close` does makes the delay expensive: it pops the conversation from
`_conversations` (releasing a concurrency lane against
`CCI_MAX_CONCURRENT_CONVERSATIONS`), unregisters the MCP bridge, and awaits
`session.aclose()` — killing the `claude` subprocess, ~300 MB of anonymous
memory. Deferring it means completed turns hold their lane and their subprocess.

The existing coverage misses this because `test_run_turn_text_then_done` and
`test_run_turn_error_closes` drain with `[c async for c in ...]`, a full drain
that *does* resume past the terminal yield. These tests consume the way the
route actually does.
"""

from __future__ import annotations

from app.claude_session import STREAM_CLOSED  # noqa: F401  (parity with suite)
from app.conversation import (
    CLOSED,
    Conversation,
    ConversationManager,
    DoneChunk,
    ErrorChunk,
)
from app.events import Error, TextDelta, TurnDone
from app.mcp_bridge import ConversationBridge, McpBridge


def make_settings(**over):
    from app.config import Settings

    base = dict(suspended_ttl_s=300, idle_session_ttl_s=900, gc_interval_s=30,
                request_timeout_s=30, permission_mode="bypassPermissions", port=8799,
                default_model="claude-opus-5", session_reuse=False)
    base.update(over)
    return Settings(**base)


class FakeSession:
    def __init__(self, events):
        self._events = list(events)
        self.sent_turns: list = []
        self.closed = False

    async def start(self):
        pass

    async def send_user_turn(self, content):
        self.sent_turns.append(content)

    async def next_event(self, timeout=None):
        if self._events:
            return self._events.pop(0)
        return STREAM_CLOSED

    async def aclose(self):
        self.closed = True

    @property
    def running(self):
        return not self.closed


def _wire(mgr: ConversationManager, mcp: McpBridge, events) -> Conversation:
    bridge = ConversationBridge("c1", [])
    sess = FakeSession(events)
    conv = Conversation(conv_id="c1", session=sess, bridge=bridge, model="sonnet")
    mgr._conversations["c1"] = conv
    mcp.register(bridge)
    return conv


async def _consume_like_the_route(mgr, conv):
    """Break on the terminal chunk, exactly as routes/chat.py does.

    The generator is held in a local so GC cannot finalize it behind our back —
    the route likewise keeps it alive for the life of the request. Without that,
    this test would be racing the garbage collector.
    """
    gen = mgr.run_turn(conv)
    seen = []
    async for ch in gen:
        seen.append(ch)
        if isinstance(ch, (DoneChunk, ErrorChunk)):
            break
    return gen, seen


# ── the clean-completion path ──────────────────────────────────────────────

async def test_done_chunk_releases_the_lane_before_yielding():
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings())
    conv = _wire(mgr, mcp, [
        TextDelta("hi"),
        TurnDone(stop_reason="end_turn", usage={"input_tokens": 1, "output_tokens": 2}),
    ])

    gen, seen = await _consume_like_the_route(mgr, conv)
    assert any(isinstance(c, DoneChunk) for c in seen)

    assert conv.state == CLOSED, "conversation still open after its turn finished"
    assert "c1" not in mgr._conversations, (
        "lane not released: a completed turn still counts against "
        "CCI_MAX_CONCURRENT_CONVERSATIONS"
    )
    assert conv.session.closed, "claude subprocess not reaped (~300 MB held)"
    assert mcp.get("c1") is None, "MCP bridge still registered for a closed conv"

    await gen.aclose()


# ── the error path ─────────────────────────────────────────────────────────

async def test_error_chunk_releases_the_lane_before_yielding():
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings())
    conv = _wire(mgr, mcp, [Error("boom")])

    gen, seen = await _consume_like_the_route(mgr, conv)
    assert any(isinstance(c, ErrorChunk) for c in seen)

    assert conv.state == CLOSED
    assert "c1" not in mgr._conversations
    assert conv.session.closed

    await gen.aclose()


# ── lane accounting is the thing that actually bit us ──────────────────────

async def test_sequential_turns_do_not_accumulate_lanes():
    """Ten completed turns must leave zero occupied lanes.

    With cleanup deferred to generator finalization, each completed turn holds
    its lane until GC — which is how a cap of 8 gets exhausted by traffic that
    is never more than one turn deep at a time.
    """
    mcp = McpBridge()
    mgr = ConversationManager(mcp, make_settings())
    held = []
    for i in range(10):
        bridge = ConversationBridge(f"s{i}", [])
        sess = FakeSession([TurnDone(stop_reason="end_turn", usage={})])
        conv = Conversation(conv_id=f"s{i}", session=sess, bridge=bridge, model="sonnet")
        mgr._conversations[f"s{i}"] = conv
        mcp.register(bridge)

        gen = mgr.run_turn(conv)
        async for ch in gen:
            if isinstance(ch, (DoneChunk, ErrorChunk)):
                break
        held.append(gen)  # keep alive: no GC rescue

    assert mgr._conversations == {}, (
        f"{len(mgr._conversations)} lanes still occupied after 10 completed turns"
    )
    assert mgr.lane_usage()["live"] == 0

    for g in held:
        await g.aclose()
