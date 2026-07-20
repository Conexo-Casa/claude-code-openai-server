"""Phase 3 — cross-turn session reuse (ships dark behind ``CCI_SESSION_REUSE``).

Today every fresh user turn spawns a brand-new ``claude`` subprocess and folds
the *entire* conversation back into one user message (see
:func:`app.translate.fold_conversation`). Prompt caching only covers the stable
system prefix, so that growing folded body is re-prefilled at full cost every
turn — the measured 22s→398s latency curve.

Phase 3 removes the re-fold. A stable per-conversation id supplied by hermes
(``hermes_session_id`` on the request) is mapped deterministically to a Claude
session UUID via :func:`uuid.uuid5`. On the first turn of a conversation the
subprocess is spawned with ``--session-id <uuid>`` and seeded with the folded
history; on every later turn it is spawned with ``--resume <uuid>`` and sent
**only the new user turn** — the CLI reads the prior context back from its
on-disk session store (cross-process prompt cache), so the bulk of the tokens
come back as ``cache_read`` instead of a fresh prefill.

Spike 3.0 evidence (opus-4-8, subscription): ``--resume`` + delta read 106,778
tokens from cache and prefilled a 501-token delta in ~2s, vs. the re-fold's
full-body re-prefill every turn.

Auth invariant (non-negotiable): this NEVER introduces an API key. Every turn
still short-lived-spawns the ``claude`` CLI, which reads the rotating OAuth
credential from ``~/.claude/.credentials.json`` exactly as today. Because each
turn is a fresh process, the credential is re-read every turn — there is no
long-lived process holding a stale token, so no 401-on-rotation risk.

Correctness guard: hermes rotates ``session_id`` when it compacts/rewrites
history (``_compress_context``), so the derived UUID changes at exactly the
moment the prefix diverges — a fresh session is seeded automatically and the
stale one is left to GC. Non-compaction history edits are out of scope for this
dark MVP (the primary WebUI workload appends linearly).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("cci.reuse")

# Fixed namespace so uuid5(namespace, hermes_session_id) is reproducible across
# processes and restarts. Do NOT change this constant once sessions exist on
# disk keyed by it, or every conversation re-seeds once after the change.
NAMESPACE_CCI = uuid.UUID("6f9c2e1a-3b7d-5e4a-9c21-0d8f4a2b7e61")

# Modes a fresh turn can take.
MODE_SEED = "seed"      # first contact: --session-id <uuid>, send folded history
MODE_RESUME = "resume"  # later turn: --resume <uuid>, send only the new delta
MODE_LEGACY = "legacy"  # reuse disabled / no id: today's fold-everything path


def derive_session_uuid(hermes_session_id: str) -> str:
    """Map hermes' stable per-conversation id to a deterministic Claude UUID."""
    return str(uuid.uuid5(NAMESPACE_CCI, hermes_session_id))


def project_dir_for(workdir: Union[str, Path]) -> Path:
    """Return the Claude Code session-store dir for a working directory.

    Claude Code persists sessions to
    ``~/.claude/projects/<mangled-abs-workdir>/<session-uuid>.jsonl`` where the
    mangle replaces every ``/`` (and ``.``) in the absolute path with ``-``
    (e.g. ``/home/behlers/cci-workspace`` → ``-home-behlers-cci-workspace``).
    Verified empirically on the deployment host.
    """
    abs_path = str(Path(workdir).expanduser().resolve())
    mangled = abs_path.replace("/", "-").replace(".", "-")
    return Path.home() / ".claude" / "projects" / mangled


def session_exists_on_disk(session_uuid: str, workdir: Union[str, Path]) -> bool:
    """True if Claude Code already has an on-disk session file for this UUID."""
    try:
        return (project_dir_for(workdir) / f"{session_uuid}.jsonl").is_file()
    except OSError:  # pragma: no cover - defensive (permission/FS errors)
        return False


@dataclass
class ReusePlan:
    """Decision for one fresh turn: how to spawn and what to send."""

    mode: str                       # MODE_SEED | MODE_RESUME | MODE_LEGACY
    session_uuid: Optional[str] = None

    @property
    def resume_id(self) -> Optional[str]:
        return self.session_uuid if self.mode == MODE_RESUME else None

    @property
    def assign_id(self) -> Optional[str]:
        return self.session_uuid if self.mode == MODE_SEED else None


class SessionRegistry:
    """Tracks which derived session UUIDs have been seeded, to pick seed vs.
    resume for a fresh turn.

    In-memory ``_seen`` is authoritative within a process (no false positives:
    we only add a UUID once we have spawned/seeded it). Across a restart the set
    is empty, so a disk probe backfills it — the on-disk session file is the
    durable source of truth, which is what makes reuse survive a wrapper
    restart. If both miss, we seed (the correct choice for a genuinely new
    conversation).
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def plan(
        self,
        hermes_session_id: Optional[str],
        workdir: Union[str, Path],
        *,
        enabled: bool,
    ) -> ReusePlan:
        """Decide seed/resume/legacy for a fresh turn.

        ``legacy`` (today's fold-everything behavior) is returned whenever the
        feature is off or hermes supplied no conversation id — so the feature is
        a strict superset and off-by-default is byte-for-byte the old path.
        """
        if not enabled or not hermes_session_id:
            return ReusePlan(mode=MODE_LEGACY)

        session_uuid = derive_session_uuid(hermes_session_id)
        if session_uuid in self._seen:
            return ReusePlan(mode=MODE_RESUME, session_uuid=session_uuid)
        if session_exists_on_disk(session_uuid, workdir):
            self._seen.add(session_uuid)
            logger.info("reuse: resuming on-disk session %s (hsid=%s)",
                        session_uuid, hermes_session_id)
            return ReusePlan(mode=MODE_RESUME, session_uuid=session_uuid)

        # First contact for this conversation → seed a new session under our id.
        self._seen.add(session_uuid)
        logger.info("reuse: seeding new session %s (hsid=%s)",
                    session_uuid, hermes_session_id)
        return ReusePlan(mode=MODE_SEED, session_uuid=session_uuid)

    def forget(self, session_uuid: str) -> None:
        """Drop a UUID from the seen-set (e.g. after a resume miss) so the next
        turn re-probes disk and re-seeds if needed."""
        self._seen.discard(session_uuid)
