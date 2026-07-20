"""Unit tests for Phase 3 cross-turn session reuse (app/session_reuse.py)."""

from __future__ import annotations

import app.session_reuse as sr
from app.openai_models import ChatMessage
from app.session_reuse import (
    KEY_CONTENT,
    KEY_HSID,
    MODE_LEGACY,
    MODE_RESUME,
    MODE_SEED,
    ReusePlan,
    SessionRegistry,
    content_anchor,
    derive_content_session_uuid,
    derive_session_uuid,
    project_dir_for,
)


# ── message-list builders (content-lane fixtures) ───────────────────────────


def _u(text):
    return ChatMessage(role="user", content=text)


def _a(text):
    return ChatMessage(role="assistant", content=text)


def _turn(*texts):
    """Build a conversation from alternating user/assistant texts."""
    msgs = []
    for i, t in enumerate(texts):
        msgs.append(_u(t) if i % 2 == 0 else _a(t))
    return msgs


# ── uuid derivation ─────────────────────────────────────────────────────────


def test_derive_is_deterministic_and_distinct():
    a1 = derive_session_uuid("hermes-conv-A")
    a2 = derive_session_uuid("hermes-conv-A")
    b = derive_session_uuid("hermes-conv-B")
    assert a1 == a2                # stable across calls (and processes/restarts)
    assert a1 != b                 # distinct conversations → distinct sessions
    # Valid UUID string form.
    assert len(a1) == 36 and a1.count("-") == 4


# ── project-dir mangle ──────────────────────────────────────────────────────


def test_project_dir_mangle_matches_claude_code():
    p = project_dir_for("/home/behlers/cci-workspace")
    assert p.name == "-home-behlers-cci-workspace"
    assert p.parent.name == "projects"


def test_project_dir_mangle_replaces_dots():
    p = project_dir_for("/srv/app.v2/work")
    assert p.name == "-srv-app-v2-work"


# ── ReusePlan id routing ────────────────────────────────────────────────────


def test_reuse_plan_id_properties():
    seed = ReusePlan(mode=MODE_SEED, session_uuid="U")
    assert seed.assign_id == "U" and seed.resume_id is None
    resume = ReusePlan(mode=MODE_RESUME, session_uuid="U")
    assert resume.resume_id == "U" and resume.assign_id is None
    legacy = ReusePlan(mode=MODE_LEGACY)
    assert legacy.resume_id is None and legacy.assign_id is None


# ── registry decisions ──────────────────────────────────────────────────────


def test_disabled_is_legacy():
    reg = SessionRegistry()
    plan = reg.plan("hsid", [_u("hi")], "/tmp/wd", enabled=False)
    assert plan.mode == MODE_LEGACY and plan.session_uuid is None


def test_missing_id_and_no_convo_is_legacy():
    reg = SessionRegistry()
    plan = reg.plan(None, [], "/tmp/wd", enabled=True)
    assert plan.mode == MODE_LEGACY


def test_first_contact_seeds_then_resumes(monkeypatch):
    # No on-disk session anywhere.
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    first = reg.plan("hsid-1", [_u("hi")], "/tmp/wd", enabled=True)
    assert first.mode == MODE_SEED and first.key_source == KEY_HSID
    assert first.session_uuid == derive_session_uuid("hsid-1")
    # Same conversation again → resume (now in the seen-set).
    second = reg.plan("hsid-1", [_u("hi")], "/tmp/wd", enabled=True)
    assert second.mode == MODE_RESUME
    assert second.session_uuid == first.session_uuid


def test_disk_backfill_resumes_after_restart(monkeypatch):
    # Fresh registry (as after a restart) but the session file is on disk →
    # resume without re-seeding.
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: True)
    reg = SessionRegistry()
    plan = reg.plan("hsid-restart", [_u("hi")], "/tmp/wd", enabled=True)
    assert plan.mode == MODE_RESUME


def test_forget_forces_reseed(monkeypatch):
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    p1 = reg.plan("hsid-x", [_u("hi")], "/tmp/wd", enabled=True)
    assert p1.mode == MODE_SEED
    reg.forget(p1.session_uuid)
    # After forget, disk still says no → seed again (not resume a nonexistent).
    p2 = reg.plan("hsid-x", [_u("hi")], "/tmp/wd", enabled=True)
    assert p2.mode == MODE_SEED


def test_distinct_conversations_get_distinct_sessions(monkeypatch):
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    a = reg.plan("conv-a", [_u("hi")], "/tmp/wd", enabled=True)
    b = reg.plan("conv-b", [_u("hi")], "/tmp/wd", enabled=True)
    assert a.session_uuid != b.session_uuid


# ── content lane (no hermes_session_id on the wire) ─────────────────────────


def test_content_anchor_is_first_user_text():
    assert content_anchor([_u("  Hello there  "), _a("hi")]) == "Hello there"
    # Empty / non-user leading message → no anchor.
    assert content_anchor([]) is None
    assert content_anchor([_a("assistant first")]) is None
    assert content_anchor([_u("   ")]) is None


def test_content_seed_then_resume_across_turns(monkeypatch):
    # WebUI-style path: no hsid, so keying falls to content. First message is
    # byte-stable across turns, so turn 2+ resumes the seeded session.
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    t1 = reg.plan(None, [_u("start the task")], "/tmp/wd", enabled=True)
    assert t1.mode == MODE_SEED and t1.key_source == KEY_CONTENT
    assert t1.session_uuid == derive_content_session_uuid("start the task")
    # Turn 2 (first reply present) → resume, guard captured.
    t2 = reg.plan(None, _turn("start the task", "ok", "next"), "/tmp/wd", enabled=True)
    assert t2.mode == MODE_RESUME and t2.session_uuid == t1.session_uuid
    # Turn 3, same first reply → still resume.
    t3 = reg.plan(None, _turn("start the task", "ok", "next", "more", "again"),
                  "/tmp/wd", enabled=True)
    assert t3.mode == MODE_RESUME and t3.session_uuid == t1.session_uuid


def test_content_opening_turn_collision_is_legacy(monkeypatch):
    # A *different* conversation opens with the identical first line while the
    # first is already seeded → must NOT resume into it.
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    a1 = reg.plan(None, [_u("status")], "/tmp/wd", enabled=True)
    assert a1.mode == MODE_SEED
    # Second chat, same opening line, single message → ambiguous → legacy.
    b1 = reg.plan(None, [_u("status")], "/tmp/wd", enabled=True)
    assert b1.mode == MODE_LEGACY


def test_content_divergence_guard_blocks_merge(monkeypatch):
    # Two chats share the opening line but diverge at the first reply → the
    # second must fall back to legacy, never resume into the first's session.
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    # Chat A seeds, then establishes its guard (first reply "AAA").
    reg.plan(None, [_u("status")], "/tmp/wd", enabled=True)
    a2 = reg.plan(None, _turn("status", "AAA", "go on"), "/tmp/wd", enabled=True)
    assert a2.mode == MODE_RESUME
    # Chat B, same opening line but a different first reply ("BBB") → collision.
    b2 = reg.plan(None, _turn("status", "BBB", "go on"), "/tmp/wd", enabled=True)
    assert b2.mode == MODE_LEGACY


def test_content_disk_backfill_resumes_after_restart(monkeypatch):
    # Fresh registry (post-restart), continuing turn, session file on disk →
    # resume on the content lane without re-seeding.
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: True)
    reg = SessionRegistry()
    plan = reg.plan(None, _turn("resume me", "ok", "again"), "/tmp/wd", enabled=True)
    assert plan.mode == MODE_RESUME and plan.key_source == KEY_CONTENT


def test_hsid_preferred_over_content(monkeypatch):
    # When both an hsid and content are available, the hsid fast lane wins and
    # keys on the hsid (not the content anchor).
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    p = reg.plan("the-hsid", [_u("some opening")], "/tmp/wd", enabled=True)
    assert p.key_source == KEY_HSID
    assert p.session_uuid == derive_session_uuid("the-hsid")


# ── ClaudeSession arg wiring ────────────────────────────────────────────────


def _args(**kw):
    from app.claude_session import ClaudeSession

    sess = ClaudeSession(
        claude_bin="claude", model="claude-opus-4-8",
        permission_mode="bypassPermissions", workdir="/tmp/wd", **kw,
    )
    return sess._build_args()


def test_build_args_default_has_no_session_flags():
    args = _args()
    assert "--resume" not in args and "--session-id" not in args


def test_build_args_resume_flag():
    args = _args(resume_session_id="abc-123")
    assert args[args.index("--resume") + 1] == "abc-123"
    assert "--session-id" not in args


def test_build_args_assign_flag():
    args = _args(assign_session_id="def-456")
    assert args[args.index("--session-id") + 1] == "def-456"
    assert "--resume" not in args


def test_build_args_resume_wins_when_both_set():
    args = _args(resume_session_id="r-1", assign_session_id="a-1")
    assert "--resume" in args and "--session-id" not in args
