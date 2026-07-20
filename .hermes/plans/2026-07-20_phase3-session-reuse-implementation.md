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

---

## Addendum (2026-07-20) — content-fingerprint fallback for surfaces with no `hermes_session_id`

### Why

Live activation exposed a gap the original keying assumed away. The `hermes_session_id`
fast-lane requires Hermes to attach the id on the wire. It does for the gateway/CLI agent
paths — but the **WebUI** reaches the model through a separate multi-hop path
(`hermes-webui` → runtime_adapter → runner_client → runner → `AIAgent`), and on the deployed
June build `agent.session_id` is **not** threaded onto the model call there. Result: the wrapper
received `hermes_session_id=None` and every turn logged `reuse=legacy` even with the flag on and
the emit-patch live. (Also uncovered en route: the deployment's provider is named
`claude-code-server`, not the literal `custom`, so the Hermes-side guard had to match the real
provider name. Fixed in the maintained host-patch overlay, tracked in the ops repo.)

Chasing the id across the WebUI's internal hops would be fragile (separate `/opt/hermes-webui`
code tree, uncertain durability across updates) and would only fix one surface.

### Design — key on the conversation itself when no id is present

The server already receives the full conversation every turn (stateless OpenAI endpoint), so it
can derive its own stable key. Preference order, in `SessionRegistry.plan()`:

1. **`hermes_session_id`** when present — unchanged fast lane, unique per conversation, no guard.
2. **Content anchor** otherwise — `uuid5(NAMESPACE, "cci-content:" + first_user_message_text)`.
   The first user message is byte-stable across every turn of a linear conversation and is
   independent of any per-turn drift in the system prompt, so keying on it (rather than
   `system + first_user`) is drift-proof — it cannot silently lose reuse if Hermes varies the
   system block. Mirrors Hermes' own `_derive_chat_session_id` (`api-<digest>`) for OpenAI-compat
   frontends.

**Divergence guard (makes the fallback safe).** Two *different* conversations that open with the
byte-identical first line derive the same key. To never merge them:
- Opening turn (history length 1): if the derived session is already seeded, treat it as a
  different conversation and fall back to `legacy` (a throwaway fold — for one message, just that
  message). We cannot yet tell two identical opens apart, so we never resume into the other one.
- Continuing turn: fingerprint the first assistant reply (message index 1) at seed time and
  re-check it on every resume. On mismatch, a different conversation has collided on the anchor →
  fall back to `legacy` instead of resuming. Net: correctness is always preserved; the cache win
  goes to the first conversation with a given opening line, and any collider runs exactly like
  today. (Relevant here because this user often opens chats with short generic lines like
  "status".)

Compaction still re-seeds cleanly on both lanes: hsid rotates on `_compress_context`; the content
anchor changes when Hermes rewrites the leading messages.

### Verification (content lane, no `hermes_session_id` on the wire)

- Unit: `tests/test_session_reuse.py` now 20 tests (seed→resume across turns, opening-turn
  collision → legacy, divergence-guard → legacy, disk-backfill resume, hsid-preferred-over-content).
  Full suite **115 passing**, green both with and without an ambient `CCI_SESSION_REUSE` in the env
  (warm-pool adoption tests pin `session_reuse=False`, since a Phase-3 seed/resume correctly
  bypasses the generic warm pool).
- E2E, throwaway server on :8801, flag ON, **no `hermes_session_id` sent**, isolated workdir,
  production :8787 untouched:
  - Turn 1 → `reuse[content]: seed`; Turn 2 (same first user message) → `reuse[content]: resume`.
  - Turn 2's delta did **not** restate the planted secret; the answer recalled it correctly
    (`ZEPHYR-2291`) — proof the session resumed and read prior context back from cache.
  - Cost $0.898 → $0.0438 (~20×) on the subscription path (`cost_usd` reported, no API key).

### Files changed (addendum)

- `app/session_reuse.py` — content anchor + divergence-guard registry; `plan()` now takes `convo`.
- `app/routes/chat.py`, `app/conversation.py` — pass `convo` into `plan()`; log `key_source`.
- `tests/test_session_reuse.py` — content-lane coverage; `tests/test_warmpool.py` — pin reuse off.
