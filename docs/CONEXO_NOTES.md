# Conexo-Casa deployment notes

This is Conexo-Casa's maintained fork of
[`schmarta/claude-code-openai-server`](https://github.com/schmarta/claude-code-openai-server).
It runs on the `hermes` host as the OpenAI-compatible backend for the Hermes agent
(`hermes-api.conexo.casa`), on the Claude **subscription** — no `ANTHROPIC_API_KEY`.

## Why we forked

Phase 3 (cross-turn session reuse) replaces the per-turn full-history re-fold with
`claude --resume <session_id>`, so each turn re-reads prior context from cache instead
of re-prefilling it. See `docs/phase3-cross-turn-session-reuse-scope.md` (private ops
repo) and `.hermes/plans/2026-07-20_phase3-session-reuse-implementation.md`.

## Runtime feature flag

Phase 3 is **dark by default**. Behaviour is identical to upstream unless enabled:

    CCI_SESSION_REUSE=1     # opt in to --resume cross-turn reuse (default: off)

The Hermes side must also send a stable per-conversation id as `hermes_session_id`
in the request body. On the deployed `hermes-suite` container that is provided by the
overlay image (see the private `hermes-conexo-infra` ops repo). Without it, the wrapper
falls back to content-fingerprint keying, and with the flag off it re-folds as upstream.

Auth invariant: no code path introduces an API key. Every turn spawns a short-lived
`claude` CLI that re-reads the rotating OAuth credential from `~/.claude/.credentials.json`.

## Deployment on the hermes host

- systemd unit `claude-code-openai-server.service`, `User=behlers`,
  `WorkingDirectory=/opt/containers/claude-code-openai-server`, uvicorn on `:8787`.
- The working copy is deployed by **git checkout**, not a container image.

### Point the deployment at this fork (one-time)

    sudo git -C /opt/containers/claude-code-openai-server remote set-url origin \
      https://github.com/Conexo-Casa/claude-code-openai-server.git
    sudo git -C /opt/containers/claude-code-openai-server fetch origin
    sudo git -C /opt/containers/claude-code-openai-server checkout main
    sudo git -C /opt/containers/claude-code-openai-server pull --ff-only
    sudo systemctl restart claude-code-openai-server

### Sync upstream changes into the fork

    git remote add upstream https://github.com/schmarta/claude-code-openai-server.git   # once
    git fetch upstream
    git checkout main && git merge upstream/main      # resolve, keep Phase 3
    git push origin main
