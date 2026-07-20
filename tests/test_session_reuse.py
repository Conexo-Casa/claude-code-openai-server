"""Unit tests for Phase 3 cross-turn session reuse (app/session_reuse.py)."""

from __future__ import annotations

import app.session_reuse as sr
from app.session_reuse import (
    MODE_LEGACY,
    MODE_RESUME,
    MODE_SEED,
    ReusePlan,
    SessionRegistry,
    derive_session_uuid,
    project_dir_for,
)


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
    plan = reg.plan("hsid", "/tmp/wd", enabled=False)
    assert plan.mode == MODE_LEGACY and plan.session_uuid is None


def test_missing_id_is_legacy():
    reg = SessionRegistry()
    plan = reg.plan(None, "/tmp/wd", enabled=True)
    assert plan.mode == MODE_LEGACY


def test_first_contact_seeds_then_resumes(monkeypatch):
    # No on-disk session anywhere.
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    first = reg.plan("hsid-1", "/tmp/wd", enabled=True)
    assert first.mode == MODE_SEED
    assert first.session_uuid == derive_session_uuid("hsid-1")
    # Same conversation again → resume (now in the seen-set).
    second = reg.plan("hsid-1", "/tmp/wd", enabled=True)
    assert second.mode == MODE_RESUME
    assert second.session_uuid == first.session_uuid


def test_disk_backfill_resumes_after_restart(monkeypatch):
    # Fresh registry (as after a restart) but the session file is on disk →
    # resume without re-seeding.
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: True)
    reg = SessionRegistry()
    plan = reg.plan("hsid-restart", "/tmp/wd", enabled=True)
    assert plan.mode == MODE_RESUME


def test_forget_forces_reseed(monkeypatch):
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    p1 = reg.plan("hsid-x", "/tmp/wd", enabled=True)
    assert p1.mode == MODE_SEED
    reg.forget(p1.session_uuid)
    # After forget, disk still says no → seed again (not resume a nonexistent).
    p2 = reg.plan("hsid-x", "/tmp/wd", enabled=True)
    assert p2.mode == MODE_SEED


def test_distinct_conversations_get_distinct_sessions(monkeypatch):
    monkeypatch.setattr(sr, "session_exists_on_disk", lambda u, w: False)
    reg = SessionRegistry()
    a = reg.plan("conv-a", "/tmp/wd", enabled=True)
    b = reg.plan("conv-b", "/tmp/wd", enabled=True)
    assert a.session_uuid != b.session_uuid


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
