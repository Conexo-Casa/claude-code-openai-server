"""Runtime configuration (env-driven, prefix ``CCI_``) and model-alias resolution.

All knobs are read from the environment with a ``CCI_`` prefix (e.g.
``CCI_PORT=9000``). Defaults are chosen so the server runs out of the box against
a local ``claude login`` session with no API key.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model aliases the `claude` CLI accepts directly on `--model`. Anything in this
# set, or any id beginning with "claude", is passed through verbatim; everything
# else falls back to `default_model`. The CLI itself resolves an alias like
# "opus" to the current concrete id, so we never hard-code dated ids here.
KNOWN_MODEL_ALIASES = {
    "opus",
    "sonnet",
    "haiku",
    "fable",
    "opusplan",
    "default",
}


class Settings(BaseSettings):
    """Server settings. Override any field via a ``CCI_<FIELD>`` environment var."""

    model_config = SettingsConfigDict(env_prefix="CCI_", extra="ignore")

    # ── HTTP server ──────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8787

    # ── Auth ─────────────────────────────────────────────────────────────────
    # Optional bearer token. When set, every /v1 request must carry
    # `Authorization: Bearer <api_key>`. Left unset, the server is open — which
    # is only safe on a loopback bind (see the startup guard in main.create_app).
    api_key: str | None = None

    # ── Claude CLI ───────────────────────────────────────────────────────────
    claude_bin: str = "claude"
    default_model: str = "claude-opus-4-8"
    default_effort: str | None = None
    permission_mode: str = "bypassPermissions"
    # Force the CLI to inject tool schemas directly rather than behind a
    # tool-search indirection, so hermes' functions are always visible.
    enable_tool_search: bool = False

    # ── "Bare model" mode ──────────────────────────────────────────────────—
    # When true, the spawned claude is stripped of its Claude Code identity and
    # native tooling so it behaves as a plain model fronted by hermes:
    #   * the request's system message REPLACES claude's default system prompt
    #     (--system-prompt) instead of being appended,
    #   * --exclude-dynamic-system-prompt-sections drops the env/git/identity
    #     context blocks the CLI normally injects,
    #   * --tools "" removes every built-in tool, leaving only the hermes MCP
    #     tools delivered via --mcp-config.
    # Set CCI_BARE_MODEL_MODE=false to restore the legacy append+full-tools behavior.
    bare_model_mode: bool = True
    # Fallback system prompt used only in bare mode when a request carries no
    # system message of its own (hermes normally always sends one).
    bare_model_system_prompt: str = "You are a helpful AI assistant."

    # ── Output formatting ──────────────────────────────────────────────────—
    # Rewrite Markdown pipe tables to fenced monospace ASCII so they render in
    # clients (e.g. Discord) that don't support Markdown tables.
    flatten_markdown_tables: bool = True

    # ── Workspace ──────────────────────────────────────────────────────────—
    default_workdir: Path = Path("~/cci-workspace").expanduser()
    # Per-request `workdir` overrides must resolve under one of these roots.
    # Empty => only `default_workdir` is allowed.
    allowed_workdir_roots: list[str] = []

    # ── MCP bridge ─────────────────────────────────────────────────────────—
    mcp_path_prefix: str = "/mcp"

    # ── Lifecycle / timeouts (seconds) ─────────────────────────────────────—
    request_timeout_s: int = 600
    suspended_ttl_s: int = 300
    idle_session_ttl_s: int = 900
    gc_interval_s: int = 30

    # ── Concurrency ceiling ────────────────────────────────────────────────—
    # Hard cap on live conversations. Each one owns a `claude` CLI subprocess
    # holding ~300 MB RSS, and without a ceiling the only limit is physical RAM:
    # on a 7.7 GB host with ~1.8 GB committed elsewhere, ~19 lanes exhaust memory
    # and the OOM killer takes the whole server rather than one request. 12 keeps
    # ~2.4 GB headroom while sitting well clear of observed peak use (7 lanes
    # during one interactive session). Turns beyond the cap get an OpenAI-shaped
    # 429 and may retry; continuations of an already-live conversation are never
    # refused, since they reuse a subprocess rather than spawning one. 0 disables.
    max_concurrent_conversations: int = 12

    # ── Logging ────────────────────────────────────────────────────────────—
    log_level: str = "INFO"
    # When true, emit per-turn latency metrics (spawn_ms / ttft_ms / total_ms /
    # tok_per_s) to the "cci.timing" logger at INFO. Default off so normal
    # operation pays nothing; scripts/bench.py sets it on the throwaway port.
    timing_log: bool = False

    # ── Warm subprocess pool ───────────────────────────────────────────────—
    # Number of pre-spawned, idle `claude` subprocesses kept ready to adopt on a
    # fresh tool/autonomous turn, eliminating cold-start latency. 0 disables the
    # pool entirely (ships dark). Each pooled proc holds a pre-minted conv_id and
    # an empty bridge whose tools are late-bound on acquire. See app/warmpool.py.
    warm_pool_size: int = 0

    # ── Cross-turn session reuse (Phase 3) ─────────────────────────────────—
    # When true, a fresh turn carrying hermes' stable ``hermes_session_id`` is
    # mapped to a deterministic Claude session UUID: the first turn seeds it with
    # ``--session-id`` and later turns ``--resume`` it, sending only the new user
    # turn instead of re-folding the whole transcript. This lets the CLI read the
    # prior context from its on-disk store as prompt-cache hits rather than
    # re-prefilling it every turn. Ships dark (off) — with it off, or when a
    # request carries no ``hermes_session_id``, behavior is byte-for-byte the
    # legacy fold-everything path. See app/session_reuse.py. Preserves the
    # subscription-auth invariant: every turn still spawns the CLI, which reads
    # the OAuth credential; no API key is ever introduced.
    session_reuse: bool = False

    @field_validator("default_workdir", mode="before")
    @classmethod
    def _expand_workdir(cls, v: object) -> object:
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    def is_loopback_host(self) -> bool:
        """True when the bind host only accepts connections from this machine."""
        h = self.host.strip().lower()
        return h in {"localhost", "::1", "0:0:0:0:0:0:0:1"} or h.startswith("127.")

    def mcp_dial_host(self) -> str:
        """Host the spawned ``claude`` dials to reach our per-conversation MCP
        endpoint. This must be an address we actually bound: hardcoding loopback
        breaks every session when the server is bound to a specific non-loopback
        address (e.g. the docker bridge IP) — the dial is refused, the "hermes"
        MCP server fails to load, and --strict-mcp-config then leaves the child
        with no hermes tools at all. A wildcard bind is the one case where
        loopback is the right dial target.
        """
        h = self.host.strip()
        if h in {"0.0.0.0", "::", "*", ""}:
            return "127.0.0.1"
        return h

    def resolved_workdir_roots(self) -> list[Path]:
        """The set of allowed workspace roots as resolved absolute paths."""
        roots = [self.default_workdir]
        roots.extend(Path(r).expanduser() for r in self.allowed_workdir_roots)
        return [p.resolve() for p in roots]

    def resolve_workdir(self, requested: str | None) -> Path:
        """Resolve and validate a per-request workdir override.

        Returns ``default_workdir`` when no override is given. Raises
        ``ValueError`` if the override escapes every allowed root (traversal
        guard).
        """
        if not requested:
            return self.default_workdir
        candidate = Path(requested).expanduser().resolve()
        for root in self.resolved_workdir_roots():
            if candidate == root or root in candidate.parents:
                return candidate
        raise ValueError(
            f"workdir {requested!r} is not under an allowed root "
            f"({[str(r) for r in self.resolved_workdir_roots()]})"
        )


def resolve_model(requested: str | None, settings: Settings) -> str:
    """Map an OpenAI ``model`` field to a value the `claude` CLI accepts.

    Known aliases (opus/sonnet/haiku/…) and any ``claude*`` id pass through;
    anything else (including ``None`` or empty) falls back to the default model.
    """
    if not requested:
        return settings.default_model
    r = requested.strip()
    if r in KNOWN_MODEL_ALIASES or r.lower().startswith("claude"):
        return r
    return settings.default_model


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()


def ensure_workdir(path: Path) -> Path:
    """Create the workspace dir if missing; return it. Best-effort."""
    os.makedirs(path, exist_ok=True)
    return path
