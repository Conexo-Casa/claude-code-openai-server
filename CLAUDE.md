# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Conexo-Casa's maintained fork of [`schmarta/claude-code-openai-server`](https://github.com/schmarta/claude-code-openai-server):
an OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/models`) that drives the
`claude` CLI as a subprocess over its bidirectional `stream-json` protocol.

`README.md` documents the HTTP surface, every `CCI_*` setting, and the security model —
consult it rather than duplicating it here. `docs/CONEXO_NOTES.md` covers the fork rationale
and upstream-sync procedure. This file covers what spans several files, plus the invariants
that are expensive to rediscover.

## Commands

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"   # requires-python >=3.11

.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8787   # run
.venv/bin/claude-code-interface                                          # run, uvloop+httptools pinned

.venv/bin/python -m pytest -q                                  # unit tests; no claude CLI needed
.venv/bin/python -m pytest tests/test_translate.py -q          # one file
.venv/bin/python -m pytest tests/test_translate.py::test_name -q   # one test
.venv/bin/python -m pytest -q -k session_reuse                 # by keyword
```

The deployed `.venv` on the `hermes` host runs Python 3.14, not the 3.11 the README's
install snippet names; `requires-python` is `>=3.11`, so match the existing venv rather than
recreating it at 3.11.

`pyproject.toml` already sets `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed) and
`testpaths = ["tests"]`. The root `conftest.py` prepends the repo root to `sys.path` — an
editable PEP 660 install otherwise resolves `app` as a namespace package and skips
`app/__init__.py`; don't remove it.

Live checks need a **running server** and a logged-in `claude`:

```bash
.venv/bin/python tests/scripts/e2e_autonomous.py   # text turn
.venv/bin/python tests/scripts/e2e_tool.py         # full tool loop
.venv/bin/python tests/scripts/smoke_session.py
.venv/bin/python scripts/bench.py --port 8799 --iters 3 --out bench.json
```

`scripts/bench.py` starts its own throwaway uvicorn — it never touches a running server — and
sets `CCI_PORT` (not just `--port`) so the per-conversation MCP callback URL self-matches.
Compare against the committed `bench-baseline.json`.

## Architecture

Requests fork in `app/routes/chat.py` on one condition — whether the client sent `tools`:

**Autonomous path** (no `tools`). History is folded into a single user turn
(`translate.fold_conversation`), a fresh subprocess streams the reply, and the turn ends. No
state survives.

**Tool path** (`_handle_tools` → `app/conversation.py`). A *conversation* owns one live
`claude` subprocess plus a `ConversationBridge`, and outlives the HTTP request:

```
RUNNING ──tool calls──▶ SUSPENDED ──(next request carries results)──▶ RUNNING
        └─────────── clean result ───────────▶ CLOSED
```

The subprocess **blocks inside the MCP call** awaiting a Future while the server returns a
normal OpenAI `tool_calls` response. The next request resolves that Future by `tool_call_id`
and the same subprocess resumes — so one multi-step client tool loop is one conversation and
one process, matched by minted ids rather than by hashing history.

Module map — most carry a detailed docstring worth reading before changing behaviour. One
exception: `app/routes/chat.py`'s docstring is stale (it claims `tools` is "logged and ignored",
but the tool path has since landed at line 78) — trust the code there.

| File | Role |
|---|---|
| `app/routes/chat.py` | Path fork, SSE vs collected assembly, concurrency ceiling |
| `app/conversation.py` | Conversation lifecycle, continuation matching, GC |
| `app/claude_session.py` | `stream-json` subprocess driver (asyncio port of wisp's `claude/mod.rs`) |
| `app/mcp_bridge.py` | One MCP server for all conversations; `conv_id` via `ContextVar` |
| `app/session_reuse.py` | Phase 3 `--resume` cross-turn reuse + divergence guard |
| `app/translate.py` | OpenAI ⇄ Claude message folding, system split, image anchoring |
| `app/events.py` | JSONL line → typed `ChatEvent` |
| `app/warmpool.py` | Pre-spawned idle procs (single signature at a time) |
| `app/textfilter.py` | Markdown pipe-table flattening, newline-seam handling |
| `app/routes/compat.py` | Ollama / llama.cpp probe shims |

## Invariants

These are load-bearing; each has already cost a debugging session.

- **Subscription auth, no API key.** Every turn spawns a short-lived `claude` that re-reads
  the rotating OAuth credential from `~/.claude/.credentials.json`. No code path may introduce
  an `ANTHROPIC_API_KEY`. This is why `HOME` is set in the systemd unit rather than the env file.
- **`control_response` shape.** `request_id` must nest *inside* the `response` object, never at
  top level, or the turn stalls forever (`app/claude_session.py`).
- **`mcp>=1.2,<2`.** The upper bound is deliberate: mcp 2.0 removes the low-level decorator API
  (`@server.list_tools()`, `@server.call_tool(validate_input=False)`) this server is built on.
  An unbounded range lets a fresh resolve pick 2.x and fail at import — which, for a systemd
  unit, means the service simply stops starting.
- **`/mcp` carries no bearer token — an accepted risk, not a safety property.** The original
  "reached only by the local subprocess over loopback" justification does *not* hold on every
  deployment: bound to a docker-bridge address, `/mcp` is reachable by anything else on that
  bridge. Read the comment in `app/main.py:create_app` and commit `2ce2c40` before changing it.
- **Non-loopback bind refuses to start without `CCI_API_KEY`.** The server drives Claude with
  `permission_mode=bypassPermissions`, so an open bind with no token is remote code execution.
- **Concurrency cap ↔ memory ceiling.** `max_concurrent_conversations` decides how many `claude`
  children can exist; the unit's `MemoryMax` covers the whole cgroup (uvicorn + every child, at
  ~285–333 MB per live lane). Re-measure both together before changing either, and check
  `memory.events`' `max` counter — cgroup pressure here is **silent**, showing up as a reclaim
  tax with no OOM kill and nothing in the journal.

## Deployment on the `hermes` host

Runs as a **system** unit (`/etc/systemd/system/claude-code-openai-server.service`,
`User=behlers`, uvicorn on `:8787`), deployed by **git checkout** — not a container image and
not the *user* unit the README's example shows. There is no `--reload`; edits on disk do
nothing until the unit restarts.

```bash
sudo systemctl restart claude-code-openai-server
systemctl status claude-code-openai-server
journalctl -u claude-code-openai-server -f
```

Config comes from `/opt/containers/env/claude-code-openai-server.env` (0640, holds
`CCI_API_KEY` — don't echo it). It is deliberately **not** written as `-EnvironmentFile=`: if
it goes missing the unit must fail, because an unset `CCI_HOST` expands to `--host ` → `0.0.0.0`.

This repo also ships its own copy of the unit at `claude-code-openai-server.service`. It is a
reference copy, **not** what systemd reads, and it drifts from the installed file. Diff the two
before trusting either, and sync deliberately after editing the installed one.

Phase 3 session reuse is gated by `CCI_SESSION_REUSE` and needs hermes to send a stable
`hermes_session_id` in the request body; without it, **tooled** requests fall back to
content-fingerprint keying with a divergence guard, and tool-less (autonomous) requests
run legacy — see `SessionRegistry.plan`'s `has_tools` note. `CCI_MAX_CONCURRENT_CONVERSATIONS` and
`CCI_SESSION_REUSE` are live in the deployed env file but absent from the README's config
table — `app/config.py` is the source of truth for defaults.
