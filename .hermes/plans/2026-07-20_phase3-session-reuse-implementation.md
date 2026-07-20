# Phase 3 — Cross-turn session reuse (implementation)

Status: **implemented, behind dark flag `CCI_SESSION_REUSE` (default off)**. Follow-up to
`2026-06-23_optimize-cci-server-latency.md` (Phase 3 was deferred there as highest blast radius).

## Problem recap

The server is stateless on the wire (OpenAI protocol). Every turn re-folds the entire
conversation history into one `"Conversation so far: …"` user message and sends it to a freshly
spawned `claude` subprocess. Phase 4 caching only covers the **system prefix**; the folded body
changes bytes every turn, so it is re-prefilled at full cost. That is the measured 22s→398s
latency curve as context grows.

## Mechanism (proven by spike 3.0, four arms)

Cache reuse is driven by **prefix byte-stability**, not process liveness:

| arm | what | cache_read | cache_create | api_ms | cost |
|-----|------|-----------:|-------------:|-------:|-----:|
| B | keep-alive proc, delta via stdin | 0 | 166,460 | 8796 | $2.456 |
| A | re-fold, current prod | 33,608 | 73,212 | 5318 | $0.791 |
| C | fresh proc, byte-identical prefix | 106,778 | 0 | 5327 | $0.094 |
| D | `--resume <sid>` + delta only | 106,778 | 501 | 2045 | $0.059 |

Keep-alive (the originally-recommended Option B) got **cache_read=0** — rejected. The design is
**Arm D**: `--resume <session_id>` in a fresh short-lived process, sending only the new turn.

Assign-then-resume also confirmed: the CLI honors a self-chosen `--session-id <uuid>` on turn 1
and `--resume <uuid>` reads it back on turn 2 (cache_read=106,778, delta=36, ~2.5s). Collision
on an existing id exits 1 "already in use" — a deterministic first-vs-resume signal.

## Design as built

- **Keying.** Hermes already threads a stable per-conversation `session_id` into the outbound
  request builder (`agent.session_id` → `build_kwargs(..., is_custom_provider=True)`). The Hermes
  emit-patch (companion change, tracked separately) attaches it as
  `extra_body["hermes_session_id"]`. The server derives the claude session UUID deterministically:
  `uuid5(NAMESPACE, hermes_session_id)`. No lookup table; the key is addressable across restarts.
- **First contact vs resume.** Attempt is disambiguated off the CLI: `--session-id` on a new id
  seeds; `--resume` on a known id replays. The "already in use" / "no conversation found" errors
  drive the branch, so no external first-turn state is kept.
- **Delta only.** On resume we send `turn_delta()` (the last user turn, multimodal-aware) instead
  of the full fold. Both request paths are intercepted: autonomous (`routes/chat.py`) and tool
  (`conversation.py`).
- **Prefix-divergence guard.** Hermes rotates its `session_id` on context compaction, so the
  derived UUID changes exactly when history is rewritten — a fresh claude session starts
  automatically. A prefix-hash check remains as backup for edits/branches that don't rotate.
- **Fallback.** Any anomaly (missing id, resume failure, divergence) falls back to the legacy
  full-fold path for that one turn — correct, just slow once.

## Auth invariant (non-negotiable)

The whole point of this wrapper is to use the **Claude subscription via OAuth**, not the metered
API. `--resume` preserves this by construction: each turn is still a short-lived process that
re-reads the rotating OAuth credential from `~/.claude/.credentials.json` at spawn. No
`ANTHROPIC_API_KEY` is introduced anywhere. E2E runs reported `cost_usd` on the subscription path
with the key unset. Because the process is short-lived per turn, the token-staleness/401 risk that
a long-lived process would carry does not apply — no lifetime cap needed.

## Files changed

- `app/session_reuse.py` (new) — registry, `uuid5` derive, project-dir mangle, divergence guard.
- `app/config.py` — `CCI_SESSION_REUSE` flag (default off).
- `app/openai_models.py` — optional `hermes_session_id` request field (model already `extra=ignore`).
- `app/claude_session.py` — `resume_session_id` / `assign_session_id` args wired into `_build_args`.
- `app/routes/chat.py` — autonomous path: consult registry, send delta on resume.
- `app/translate.py` — shared `turn_delta()` (multimodal-aware).
- `app/conversation.py` — tool path: same reuse logic through `create()`.
- `app/main.py` — registry created in lifespan, exposed on `app.state`.

## Verification

- Unit: 109 passing (95 existing + 14 new in `tests/test_session_reuse.py`).
- E2E, throwaway server on :8799, flag ON, subscription auth, production :8787 untouched:
  - Autonomous path: turn 2 `cache_read=47,757 / cache_create=32`, cost $0.48→$0.024 (~20×),
    ~2.7s, answer correct from resumed context.
  - Tool path: turn 2 answered correctly from context read back on resume.
- Production remained on legacy (flag unset), root-owned repo never modified.

## Rollout

Ship dark. Enable `CCI_SESSION_REUSE=true` on the systemd unit after the Hermes emit-patch lands
and one supervised session confirms cache_read reuse in the live logs. Rollback = unset the flag.
