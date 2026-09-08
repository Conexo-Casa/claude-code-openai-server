"""Phase 3 — cross-turn session reuse (ships dark behind ``CCI_SESSION_REUSE``).

Today every fresh user turn spawns a brand-new ``claude`` subprocess and folds
the *entire* conversation back into one user message (see
:func:`app.translate.fold_conversation`). Prompt caching only covers the stable
system prefix, so that growing folded body is re-prefilled at full cost every
turn — the measured 22s→398s latency curve.

Phase 3 removes the re-fold. A per-conversation key is mapped deterministically
to a Claude session UUID via :func:`uuid.uuid5`. On the first turn of a
conversation the subprocess is spawned with ``--session-id <uuid>`` and seeded
with the folded history; on every later turn it is spawned with
``--resume <uuid>`` and sent **only the new user turn** — the CLI reads the prior
context back from its on-disk session store (cross-process prompt cache), so the
bulk of the tokens come back as ``cache_read`` instead of a fresh prefill.

Spike 3.0 evidence (opus-4-8, subscription): ``--resume`` + delta read 106,778
tokens from cache and prefilled a 501-token delta in ~2s, vs. the re-fold's
full-body re-prefill every turn.

Two ways to obtain the per-conversation key, in preference order:

1. **``hermes_session_id`` (fast lane).** When hermes attaches a stable
   per-conversation id on the request, we key on it directly. It is unique per
   conversation, so there is no collision risk and no divergence guard is needed.

2. **Content fingerprint (fallback).** Some hermes surfaces (notably the WebUI,
   whose request reaches the model through a multi-hop runner path) do NOT thread
   a session id onto the wire. For those we derive a *stable* key from the
   conversation itself — the normalized text of the first user message, which is
   byte-stable across every turn of a linear append-only conversation and
   independent of any per-turn drift in the system prompt. This mirrors hermes'
   own ``_derive_chat_session_id`` (an ``api-<digest>`` content hash) for
   OpenAI-compatible frontends.

   The fallback is inherently best-effort: two *different* conversations that
   open with the byte-identical first user message derive the same key. A
   **divergence guard** makes that safe — we fingerprint the first assistant
   reply (message index 1) at seed time and re-check it on every resume; if a
   later turn's first reply differs, a different conversation has collided on the
   anchor and we fall back to a safe throwaway fold (:data:`MODE_LEGACY`) rather
   than resume into the wrong session. On the *opening* turn (history length 1)
   we cannot yet tell two identical opens apart, so if the derived session is
   already seeded we likewise fall back to legacy instead of merging. Net effect:
   correctness is always preserved; the cache win goes to the first conversation
   with a given opening line, and any collider silently runs exactly like today.

Auth invariant (non-negotiable): this NEVER introduces an API key. Every turn
still short-lived-spawns the ``claude`` CLI, which reads the rotating OAuth
credential from ``~/.claude/.credentials.json`` exactly as today. Because each
turn is a fresh process, the credential is re-read every turn — there is no
long-lived process holding a stale token, so no 401-on-rotation risk.

Correctness guard (hsid lane): hermes rotates ``session_id`` when it
compacts/rewrites history (``_compress_context``), so the derived UUID changes at
exactly the moment the prefix diverges — a fresh session is seeded automatically.
On the content lane, compaction rewrites the leading messages, which changes the
anchor and likewise re-seeds. Either way divergence re-seeds; it never merges.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from app.openai_models import ChatMessage

logger = logging.getLogger("cci.reuse")

# Fixed namespace so uuid5(namespace, key) is reproducible across processes and
# restarts. Do NOT change this constant once sessions exist on disk keyed by it,
# or every conversation re-seeds once after the change.
NAMESPACE_CCI = uuid.UUID("6f9c2e1a-3b7d-5e4a-9c21-0d8f4a2b7e61")

# Modes a fresh turn can take.
MODE_SEED = "seed"      # first contact: --session-id <uuid>, send folded history
MODE_RESUME = "resume"  # later turn: --resume <uuid>, send only the new delta
MODE_LEGACY = "legacy"  # reuse disabled / no key / collision: fold-everything path

# Key provenance (for logging / diagnostics).
KEY_HSID = "hsid"
KEY_CONTENT = "content"

# Separator between an anchor's text section and its non-text digest. ASCII unit
# separator: it does not occur in prose, so the fast path in `_anchor_repr`
# (which must stay byte-compatible with the pre-fix anchor) covers essentially
# all real text. Text that *does* contain it is routed through the structured
# form so the marker cannot be forged — see `_anchor_repr`.
SEP = "\x1f"

# Ceiling on tracked sessions. Each entry is a UUID plus a 16-char guard (~100
# bytes), so this is a small cap in absolute terms; the point is that the
# registry is bounded at all rather than growing for the process lifetime.
# Eviction only ever costs a cache hit — see SessionRegistry's docstring.
MAX_TRACKED_SESSIONS = 4096


def derive_session_uuid(hermes_session_id: str) -> str:
    """Map hermes' stable per-conversation id to a deterministic Claude UUID."""
    return str(uuid.uuid5(NAMESPACE_CCI, hermes_session_id))


def derive_content_session_uuid(anchor: str) -> str:
    """Map a content anchor to a deterministic Claude UUID.

    Namespaced with a ``cci-content:`` prefix so a content anchor can never
    collide with a raw ``hermes_session_id`` value in the same UUID space.
    """
    return str(uuid.uuid5(NAMESPACE_CCI, "cci-content:" + anchor))


def _normalize(text: str) -> str:
    return (text or "").strip()


def _anchor_repr(msg: ChatMessage) -> str:
    """Anchor representation of one message, covering NON-TEXT parts too.

    ``message_text`` deliberately drops non-text content parts (its docstring
    says so), which made this key blind to images: two different conversations
    opening with the same sentence and *different* attached images produced a
    byte-identical anchor, hence the same ``uuid5`` session UUID — and the
    content lane would resume one into the other's history.

    Backward compatibility is deliberate and load-bearing. For string content,
    and for a parts list that is entirely text, this returns exactly what the
    old ``_normalize(message_text(...))`` returned, so every session already on
    disk keeps its key and no live conversation loses reuse. Only messages that
    actually carry a non-text part get a new key — precisely the case that was
    broken.

    Non-text parts are folded into a short digest rather than embedded verbatim:
    a data-URL image can be megabytes, and this runs on every turn.
    ``sort_keys`` makes the digest independent of dict ordering.
    """
    content = msg.content
    if content is None:
        return ""
    if isinstance(content, str):
        return _normalize(content)

    texts: list[str] = []
    others: list[object] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text" \
                and isinstance(part.get("text"), str):
            texts.append(part["text"])
        else:
            others.append(part)

    text = _normalize("".join(texts))
    # Fast path: pure text with no separator byte returns exactly what the
    # pre-fix anchor did, so on-disk sessions keep their key.
    if not others and SEP not in text:
        return text

    # Text that itself contains SEP is routed through the structured form even
    # with no non-text parts. Otherwise a message whose text is literally
    # "hello<SEP>nontext:<digest>" would render byte-identical to the anchor of
    # a *different* conversation that really carries that image — a forgeable
    # key. Falling through appends the empty-parts digest, which no genuine
    # image can produce, so the two can never coincide.
    digest = hashlib.sha256(
        json.dumps(others, sort_keys=True, separators=(",", ":"), default=str)
        .encode("utf-8")
    ).hexdigest()[:16]
    return f"{text}{SEP}nontext:{digest}"


def content_anchor(convo: list[ChatMessage]) -> Optional[str]:
    """Stable per-conversation anchor: the first user message's content.

    Returns ``None`` when there is no usable anchor (empty conversation, a
    non-user leading message, or a first message with no content at all) — the
    caller then falls back to :data:`MODE_LEGACY`.

    The first user message is chosen deliberately: in a linear append-only
    conversation it never changes across turns, and it does not depend on the
    system prompt (which hermes may vary per turn), so keying on it is drift-proof
    where keying on ``system + first user`` would silently lose reuse whenever the
    system prompt drifts.

    Note a deliberate behaviour change: an *image-only* first message used to
    flatten to ``""`` and fall back to legacy. It now yields a real anchor (the
    non-text digest), so such conversations get reuse like any other — and,
    unlike before, two different image-only openings no longer collide.
    """
    if not convo or convo[0].role != "user":
        return None
    return _anchor_repr(convo[0]) or None


def _guard_fp(convo: list[ChatMessage]) -> Optional[str]:
    """Fingerprint the first assistant reply (message index 1), if present.

    Used only on the content lane to detect two different conversations that
    collided on the same opening line — they diverge at the first reply.

    Uses :func:`_anchor_repr` rather than ``message_text`` for the same reason
    the anchor does: a fingerprint that ignores non-text parts cannot detect
    divergence that lives entirely in them. For text replies (the normal case
    for an assistant turn) the two produce identical bytes, so existing guards
    still match and no live session is invalidated.
    """
    if len(convo) >= 2:
        return hashlib.sha256(
            _anchor_repr(convo[1]).encode("utf-8")
        ).hexdigest()[:16]
    return None


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
    key_source: str = ""            # KEY_HSID | KEY_CONTENT | ""

    @property
    def resume_id(self) -> Optional[str]:
        return self.session_uuid if self.mode == MODE_RESUME else None

    @property
    def assign_id(self) -> Optional[str]:
        return self.session_uuid if self.mode == MODE_SEED else None


class SessionRegistry:
    """Tracks which derived session UUIDs have been seeded, to pick seed vs.
    resume for a fresh turn, and (on the content lane) guards against two
    different conversations colliding on the same opening line.

    ``_sessions`` maps a seeded session UUID → its content divergence guard (the
    first-assistant-reply fingerprint, or ``None`` for the hsid lane / not yet
    captured). Membership is authoritative within a process, and the guard lives
    **only** in this dict — it is never persisted.

    That is why the **content lane does not survive a restart**: with
    ``_sessions`` empty, an on-disk session file proves a transcript exists but
    not whose it is, so resuming into it would be a blind merge between two
    conversations that happen to share an opening message. Such turns fall back
    to :data:`MODE_LEGACY` for the remainder of the conversation (each turn
    re-probes disk, finds the file, and folds again). Correctness over cache
    hits — decided 2026-09-08.

    The **hsid lane is unaffected**: ``hermes_session_id`` is unique per
    conversation, so there is nothing to collide and no guard is needed. A
    client that threads a session id therefore keeps reuse across restarts,
    which is the real fix for the cost this policy accepts.

    Bounded on purpose. ``_sessions`` used to grow for the life of the process —
    one entry per distinct conversation, never removed. It is now an LRU capped
    at :data:`MAX_TRACKED_SESSIONS`. Eviction is safe in one direction only, and
    that direction is the safe one: a forgotten UUID makes the next turn re-probe
    disk, which yields legacy if a file exists (never a blind resume) or a fresh
    seed if not. Eviction can cost a cache hit; it cannot cause a wrong merge.
    """

    def __init__(self, max_tracked: int = MAX_TRACKED_SESSIONS) -> None:
        # OrderedDict, not dict: entries are moved to the end on every touch so
        # eviction drops the least-recently-used conversation rather than the
        # oldest-created one (a long-running conversation must not be evicted
        # just because it started early).
        self._sessions: "OrderedDict[str, Optional[str]]" = OrderedDict()
        self._max_tracked = max(1, max_tracked)

    # ── registry bookkeeping ──────────────────────────────────────────────—

    def _remember(self, session_uuid: str, guard: Optional[str]) -> None:
        """Record/refresh a session's guard, evicting LRU entries past the cap."""
        self._sessions[session_uuid] = guard
        self._sessions.move_to_end(session_uuid)
        while len(self._sessions) > self._max_tracked:
            evicted, _ = self._sessions.popitem(last=False)
            logger.info("reuse: evicting LRU session %s (cap %d)",
                        evicted, self._max_tracked)

    def _touch(self, session_uuid: str) -> None:
        """Mark a session as recently used so the LRU keeps it."""
        if session_uuid in self._sessions:
            self._sessions.move_to_end(session_uuid)

    def plan(
        self,
        hermes_session_id: Optional[str],
        convo: list[ChatMessage],
        workdir: Union[str, Path],
        *,
        enabled: bool,
    ) -> ReusePlan:
        """Decide seed/resume/legacy for a fresh turn.

        ``legacy`` (today's fold-everything behavior) is returned whenever the
        feature is off, no key can be derived, or a content-anchor collision is
        detected — so the feature is a strict superset and off-by-default is
        byte-for-byte the old path.
        """
        if not enabled:
            return ReusePlan(mode=MODE_LEGACY)

        # ── fast lane: explicit stable id from hermes (unique per conversation)
        if hermes_session_id:
            return self._plan_hsid(derive_session_uuid(hermes_session_id), workdir)

        # ── fallback: derive a stable key from conversation content
        anchor = content_anchor(convo)
        if not anchor:
            return ReusePlan(mode=MODE_LEGACY)
        return self._plan_content(
            derive_content_session_uuid(anchor), workdir,
            history_len=len(convo), guard=_guard_fp(convo),
        )

    # ── hsid lane ─────────────────────────────────────────────────────────—

    def _plan_hsid(self, session_uuid: str, workdir: Union[str, Path]) -> ReusePlan:
        # The hsid key is unique per conversation, so an on-disk transcript can
        # only be this conversation's — disk backfill stays safe on this lane
        # (unlike the content lane, see _plan_content).
        if session_uuid in self._sessions:
            self._sessions.setdefault(session_uuid, None)
            self._touch(session_uuid)
            logger.info("reuse[hsid]: resume %s", session_uuid)
            return ReusePlan(MODE_RESUME, session_uuid, KEY_HSID)
        if session_exists_on_disk(session_uuid, workdir):
            self._remember(session_uuid, None)
            logger.info("reuse[hsid]: resume %s", session_uuid)
            return ReusePlan(MODE_RESUME, session_uuid, KEY_HSID)
        self._remember(session_uuid, None)
        logger.info("reuse[hsid]: seed %s", session_uuid)
        return ReusePlan(MODE_SEED, session_uuid, KEY_HSID)

    # ── content lane (with divergence guard) ──────────────────────────────—

    def _plan_content(
        self,
        session_uuid: str,
        workdir: Union[str, Path],
        *,
        history_len: int,
        guard: Optional[str],
    ) -> ReusePlan:
        known = session_uuid in self._sessions

        # Opening turn: a single user message. We cannot yet distinguish this
        # conversation from a *different* one that opened with the identical
        # line. If the session is already seeded, assume it is a different
        # conversation and do NOT resume into it — fall back to a safe throwaway
        # fold (which, for a 1-message history, is just that message).
        if history_len <= 1:
            if known or session_exists_on_disk(session_uuid, workdir):
                logger.info("reuse[content]: opening-turn collision on %s → legacy",
                            session_uuid)
                return ReusePlan(mode=MODE_LEGACY)
            self._remember(session_uuid, None)
            logger.info("reuse[content]: seed %s", session_uuid)
            return ReusePlan(MODE_SEED, session_uuid, KEY_CONTENT)

        # Continuing turn (history has a first reply to fingerprint).
        if not known:
            if session_exists_on_disk(session_uuid, workdir):
                # A session file exists that THIS process never seeded, so its
                # divergence guard was never captured and is not persisted
                # anywhere — we cannot tell whether that transcript belongs to
                # this conversation or to a different one that merely opened
                # with the same first message. Resuming would have been a blind
                # merge, so we never do it (policy decision, 2026-09-08).
                #
                # Legacy, not seed: MODE_SEED spawns with `--session-id <uuid>`,
                # and that UUID already has a file on disk. Legacy is also what
                # the opening-turn branch above already does for exactly this
                # situation, so the two agree.
                #
                # Deliberately do NOT record the guard here. Storing it would
                # make the next turn take the `known` path with `stored ==
                # guard` and resume into this very session, reintroducing the
                # blind merge one turn later.
                logger.info(
                    "reuse[content]: unverifiable on-disk session %s → legacy",
                    session_uuid,
                )
                return ReusePlan(mode=MODE_LEGACY)
            self._remember(session_uuid, guard)
            logger.info("reuse[content]: seed (mid-history) %s", session_uuid)
            return ReusePlan(MODE_SEED, session_uuid, KEY_CONTENT)

        stored = self._sessions.get(session_uuid)
        if stored is None:
            # First continuing turn after seeding: capture the guard, resume.
            self._remember(session_uuid, guard)
            logger.info("reuse[content]: resume (guard captured) %s", session_uuid)
            return ReusePlan(MODE_RESUME, session_uuid, KEY_CONTENT)
        if stored == guard:
            self._touch(session_uuid)
            logger.info("reuse[content]: resume %s", session_uuid)
            return ReusePlan(MODE_RESUME, session_uuid, KEY_CONTENT)

        # Divergence: a different conversation shares this opening line. Never
        # merge — fall back to a safe full fold in a throwaway subprocess.
        logger.warning("reuse[content]: anchor collision on %s (guard %s≠%s) → legacy",
                       session_uuid, guard, stored)
        return ReusePlan(mode=MODE_LEGACY)

    def forget(self, session_uuid: str) -> None:
        """Drop a UUID from the registry (e.g. after a resume miss) so the next
        turn re-probes disk and re-seeds if needed."""
        self._sessions.pop(session_uuid, None)
