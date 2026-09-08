"""Conversation ids must be unguessable, because the id *is* the access control.

`/mcp/<conv_id>` routes purely on the id and carries no bearer token — the auth
middleware in `main.create_app` gates only `/v1`. Anything that can reach our
bind address and guess a live id can `list_tools` and inject `call_tool` results
that surface to the OpenAI client as tool calls the model never emitted, which
an agent client will then execute.

The old scheme was `conv{counter}-{int(time.time())}`: a small counter plus a
wall-clock second, i.e. a few thousand candidates for a known time window. The
"only the local subprocess dials this" assumption in the original comment does
not hold for us — we bind a docker-bridge address shared with the rest of the
estate, not loopback.

These tests pin the entropy, not the format: they assert an attacker who knows
the counter and the clock still cannot reconstruct an id.
"""

from __future__ import annotations

import time

from app.conversation import ConversationManager
from app.mcp_bridge import McpBridge
from app.warmpool import WarmPool


def make_settings(**over):
    from app.config import Settings

    # session_reuse pinned off and every field set explicitly: Settings reads
    # CCI_-prefixed env vars, and a shell descended from the running service
    # inherits the production ones.
    base = dict(suspended_ttl_s=300, idle_session_ttl_s=900, gc_interval_s=30,
                request_timeout_s=30, permission_mode="bypassPermissions", port=8799,
                default_model="claude-opus-5", session_reuse=False)
    base.update(over)
    return Settings(**base)


def _mgr():
    return ConversationManager(McpBridge(), make_settings())


def _pool():
    return WarmPool(McpBridge(), make_settings(), 0)


# ── the wall-clock leak ────────────────────────────────────────────────────

def test_conv_id_does_not_embed_wall_clock():
    """The old format appended int(time.time()) verbatim."""
    cid = _mgr()._next_conv_id()
    now = int(time.time())
    for candidate in range(now - 3, now + 4):
        assert str(candidate) not in cid, (
            f"conv id {cid!r} embeds a guessable wall-clock second"
        )


def test_pool_id_does_not_embed_wall_clock():
    pid = _pool()._next_conv_id()
    now = int(time.time())
    for candidate in range(now - 3, now + 4):
        assert str(candidate) not in pid, (
            f"pool id {pid!r} embeds a guessable wall-clock second"
        )


# ── the sharpest case: same counter, same second ───────────────────────────

def test_two_managers_in_the_same_second_do_not_collide():
    """Under the old scheme both are `conv1-<same second>` — byte-identical.

    That is not merely predictable, it is a collision: two live conversations
    would share one `/mcp` route.
    """
    a, b = _mgr()._next_conv_id(), _mgr()._next_conv_id()
    assert a != b, f"two fresh managers minted the same id in one second: {a!r}"


def test_two_pools_in_the_same_second_do_not_collide():
    a, b = _pool()._next_conv_id(), _pool()._next_conv_id()
    assert a != b, f"two fresh pools minted the same id in one second: {a!r}"


# ── entropy and shape ──────────────────────────────────────────────────────

def test_conv_ids_are_unique_and_high_entropy():
    mgr = _mgr()
    ids = [mgr._next_conv_id() for _ in range(500)]
    assert len(set(ids)) == 500

    suffixes = [i.split("-", 1)[1] for i in ids]
    assert len(set(suffixes)) == 500, "suffix is not the entropy source"
    # uuid4().hex — 32 hex chars, 122 bits of randomness
    for s in suffixes:
        assert len(s) == 32 and all(c in "0123456789abcdef" for c in s), s


def test_counter_prefix_is_retained_for_log_readability():
    """Entropy must not cost us greppable ids."""
    mgr = _mgr()
    assert mgr._next_conv_id().startswith("conv1-")
    assert mgr._next_conv_id().startswith("conv2-")
    assert _pool()._next_conv_id().startswith("pool1-")


# ── why it matters: the id is the routing key ──────────────────────────────

def test_mcp_url_exposes_the_id_as_the_only_credential():
    """Documents the coupling these tests exist to protect."""
    mgr = _mgr()
    cid = mgr._next_conv_id()
    url = mgr._mcp_url(cid)
    assert url.endswith(f"/{cid}")
    assert "Authorization" not in url and "token" not in url.lower()
