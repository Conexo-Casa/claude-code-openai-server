"""The content lane never resumes into an on-disk session it cannot verify.

The divergence guard (first-assistant-reply fingerprint) lives only in
`SessionRegistry._sessions` and is never persisted. So after a restart an
on-disk session file proves a transcript exists, but not *whose* it is: a
different conversation that merely opened with the same first message derives
the same UUID. The old code resumed anyway ("disk-backfill resume"), which was
a blind merge — user B continuing into user A's transcript.

Policy, decided 2026-09-08: such turns fall back to MODE_LEGACY. Legacy rather
than seed, because MODE_SEED spawns with `--session-id <uuid>` and that UUID
already has a file on disk.

The cost is explicit and accepted: a content-lane conversation loses reuse for
the rest of its life after a restart. The hsid lane is unaffected — its key is
unique per conversation, so it keeps resuming across restarts.
"""

from __future__ import annotations

from pathlib import Path

import app.session_reuse as sr
from app.openai_models import ChatMessage
from app.session_reuse import (
    KEY_CONTENT,
    MODE_LEGACY,
    MODE_RESUME,
    MODE_SEED,
    SessionRegistry,
)

WORKDIR = "/tmp"


def u(text) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def a(text) -> ChatMessage:
    return ChatMessage(role="assistant", content=text)


def convo(n_turns: int = 2) -> list[ChatMessage]:
    """A continuing conversation: opening user turn, a reply, then a new turn."""
    msgs: list[ChatMessage] = [u("hello there"), a("hi, how can I help?")]
    if n_turns > 1:
        msgs.append(u("tell me more"))
    return msgs


def _force_on_disk(monkeypatch, exists: bool):
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda *_a, **_k: exists)


# ── the policy ─────────────────────────────────────────────────────────────

def test_unverifiable_on_disk_session_falls_back_to_legacy(monkeypatch):
    """Fresh registry (post-restart) + a session file on disk => legacy."""
    _force_on_disk(monkeypatch, True)
    reg = SessionRegistry()
    plan = reg.plan(None, convo(), WORKDIR, enabled=True)
    assert plan.mode == MODE_LEGACY, (
        "resumed into a session whose guard was never captured — blind merge"
    )


def test_legacy_fallback_does_not_record_a_guard(monkeypatch):
    """Recording the guard would make the NEXT turn resume — the hole moved.

    This is the subtle half of the fix: the pre-fix code assigned
    `_sessions[uuid] = guard` before the disk check, so even returning legacy
    would have armed a resume on the following turn.
    """
    _force_on_disk(monkeypatch, True)
    reg = SessionRegistry()
    reg.plan(None, convo(), WORKDIR, enabled=True)
    assert reg._sessions == {}, f"guard leaked into registry: {reg._sessions}"

    # And prove it: a second turn must ALSO be legacy, not resume.
    second = reg.plan(None, convo(), WORKDIR, enabled=True)
    assert second.mode == MODE_LEGACY, "next turn resumed — hole reintroduced"


def test_no_disk_file_still_seeds(monkeypatch):
    """Nothing on disk means nothing to collide with — seed as before."""
    _force_on_disk(monkeypatch, False)
    reg = SessionRegistry()
    plan = reg.plan(None, convo(), WORKDIR, enabled=True)
    assert plan.mode == MODE_SEED
    assert plan.key_source == KEY_CONTENT


# ── the in-process path is unchanged ───────────────────────────────────────

def test_same_process_seed_then_resume_still_works(monkeypatch):
    """The normal (no-restart) lifecycle must keep its cache win."""
    _force_on_disk(monkeypatch, False)
    reg = SessionRegistry()

    opening = reg.plan(None, [u("hello there")], WORKDIR, enabled=True)
    assert opening.mode == MODE_SEED

    following = reg.plan(None, convo(), WORKDIR, enabled=True)
    assert following.mode == MODE_RESUME, "in-process reuse regressed"
    assert following.session_uuid == opening.session_uuid


# ── the hsid lane keeps resuming across restarts ───────────────────────────

def test_hsid_lane_still_resumes_from_disk(monkeypatch):
    """hermes_session_id is unique per conversation: no collision, no guard."""
    _force_on_disk(monkeypatch, True)
    reg = SessionRegistry()
    plan = reg.plan("hermes-session-abc", convo(), WORKDIR, enabled=True)
    assert plan.mode == MODE_RESUME, (
        "hsid lane lost restart-survival; it has no collision risk and should "
        "not be caught by the content-lane policy"
    )


def test_hsid_lane_seeds_when_nothing_on_disk(monkeypatch):
    _force_on_disk(monkeypatch, False)
    reg = SessionRegistry()
    assert reg.plan("hermes-session-xyz", convo(), WORKDIR, enabled=True).mode == MODE_SEED


# ── the merge this prevents, end to end ────────────────────────────────────

def test_two_conversations_sharing_an_opening_line_never_merge(monkeypatch):
    """A restart must not let B continue into A's transcript.

    Same opening text, so the same derived UUID; A's file is on disk from
    before the restart; B arrives mid-history with a fresh registry.
    """
    _force_on_disk(monkeypatch, True)
    reg = SessionRegistry()  # fresh: simulates the restart

    b_plan = reg.plan(
        None,
        [u("hello there"), a("a completely different reply"), u("continue")],
        WORKDIR,
        enabled=True,
    )
    assert b_plan.mode == MODE_LEGACY
    # resume_id is a property, and returns None unless mode is RESUME — so this
    # asserts B got nothing to resume into regardless of session_uuid.
    assert b_plan.resume_id is None, "B was handed a session id to resume into"
