"""`SessionRegistry._sessions` is bounded, and eviction degrades safely.

It used to be a plain dict that grew for the life of the process — one entry per
distinct conversation, never removed. It is now an LRU capped at
`MAX_TRACKED_SESSIONS`.

The safety argument is what these tests pin, not just the arithmetic: an evicted
UUID makes the next turn take the `not known` path, which re-probes disk. That
yields MODE_LEGACY when a transcript exists (never a blind resume — see
test_disk_backfill_policy.py) or a fresh MODE_SEED when it does not. Eviction can
cost a cache hit; it must never cause a wrong merge.

LRU rather than FIFO matters: a long-running conversation must not be evicted
merely because it started early.
"""

from __future__ import annotations

import app.session_reuse as sr
from app.openai_models import ChatMessage
from app.session_reuse import (
    MAX_TRACKED_SESSIONS,
    MODE_LEGACY,
    MODE_RESUME,
    MODE_SEED,
    SessionRegistry,
)

WORKDIR = "/tmp/wd"


def u(text) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def a(text) -> ChatMessage:
    return ChatMessage(role="assistant", content=text)


def _no_disk(monkeypatch):
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda *_a, **_k: False)


def _on_disk(monkeypatch):
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda *_a, **_k: True)


# ── the bound holds ────────────────────────────────────────────────────────

def test_registry_never_exceeds_its_cap(monkeypatch):
    _no_disk(monkeypatch)
    reg = SessionRegistry(max_tracked=8)
    for i in range(200):
        reg.plan(f"hsid-{i}", [u("x")], WORKDIR, enabled=True)
    assert len(reg._sessions) == 8, f"registry grew to {len(reg._sessions)}"


def test_default_cap_is_the_module_constant():
    assert SessionRegistry()._max_tracked == MAX_TRACKED_SESSIONS


def test_cap_is_clamped_to_at_least_one():
    """A zero/negative cap would evict the entry it just wrote."""
    assert SessionRegistry(max_tracked=0)._max_tracked >= 1
    assert SessionRegistry(max_tracked=-5)._max_tracked >= 1


# ── eviction is LRU, not FIFO ──────────────────────────────────────────────

def test_recently_used_session_survives_eviction(monkeypatch):
    """The oldest-*created* entry must survive if it is still being used."""
    _no_disk(monkeypatch)
    reg = SessionRegistry(max_tracked=3)

    reg.plan("long-runner", [u("x")], WORKDIR, enabled=True)
    long_uuid = sr.derive_session_uuid("long-runner")

    # Two more conversations, then keep touching the first one between each new
    # arrival — exactly what a live long conversation looks like.
    for i in range(10):
        reg.plan("long-runner", [u("x")], WORKDIR, enabled=True)  # touch
        reg.plan(f"filler-{i}", [u("x")], WORKDIR, enabled=True)

    assert long_uuid in reg._sessions, "LRU evicted an actively-used session (FIFO?)"
    assert len(reg._sessions) == 3


def test_untouched_session_is_the_one_evicted(monkeypatch):
    _no_disk(monkeypatch)
    reg = SessionRegistry(max_tracked=2)
    reg.plan("stale-one", [u("x")], WORKDIR, enabled=True)
    stale = sr.derive_session_uuid("stale-one")

    reg.plan("keep-a", [u("x")], WORKDIR, enabled=True)
    reg.plan("keep-b", [u("x")], WORKDIR, enabled=True)

    assert stale not in reg._sessions
    assert len(reg._sessions) == 2


# ── eviction degrades safely, never into a blind resume ────────────────────

def test_eviction_with_transcript_on_disk_falls_back_to_legacy(monkeypatch):
    """Evicted + file on disk => legacy. This is the important one."""
    reg = SessionRegistry(max_tracked=1)
    _no_disk(monkeypatch)
    convo = [u("shared opening"), a("reply"), u("continue")]
    reg.plan(None, convo, WORKDIR, enabled=True)          # seeds, tracked

    # Evict it by admitting a different conversation.
    reg.plan(None, [u("other opening"), a("r"), u("c")], WORKDIR, enabled=True)

    _on_disk(monkeypatch)                                  # its transcript exists
    plan = reg.plan(None, convo, WORKDIR, enabled=True)
    assert plan.mode == MODE_LEGACY, (
        "an evicted session resumed from disk without a guard — blind merge"
    )
    assert plan.resume_id is None


def test_eviction_without_transcript_reseeds(monkeypatch):
    _no_disk(monkeypatch)
    reg = SessionRegistry(max_tracked=1)
    convo = [u("shared opening"), a("reply"), u("continue")]
    reg.plan(None, convo, WORKDIR, enabled=True)
    reg.plan(None, [u("other opening"), a("r"), u("c")], WORKDIR, enabled=True)

    plan = reg.plan(None, convo, WORKDIR, enabled=True)
    assert plan.mode == MODE_SEED


# ── the bound does not disturb normal operation ────────────────────────────

def test_in_process_seed_then_resume_still_works_under_the_cap(monkeypatch):
    # Transcript appears once seeded, as the CLI actually behaves.
    files: set[str] = set()
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: u in files)
    reg = SessionRegistry(max_tracked=64)
    opening = reg.plan("hsid-normal", [u("hello")], WORKDIR, enabled=True)
    assert opening.mode == MODE_SEED
    files.add(opening.session_uuid)
    follow = reg.plan("hsid-normal", [u("hello"), a("hi"), u("more")], WORKDIR, enabled=True)
    assert follow.mode == MODE_RESUME
    assert follow.session_uuid == opening.session_uuid


def test_forget_removes_an_entry():
    """forget() is the public drop; still unwired in app code (tracked item)."""
    reg = SessionRegistry()
    reg._remember("some-uuid", "guard")
    assert "some-uuid" in reg._sessions
    reg.forget("some-uuid")
    assert "some-uuid" not in reg._sessions
    reg.forget("some-uuid")  # idempotent
