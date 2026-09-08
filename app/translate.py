"""Translation between the OpenAI Chat Completions wire format and Claude Code.

Two directions:

* **OpenAI → Claude**: split out ``system`` messages (→ ``--append-system-prompt``)
  and fold the conversation into the content of a ``user`` turn. In autonomous
  mode a fresh subprocess handles one request, so the full history is folded into
  a single turn; the stateful continuation path (M6) sends only the latest turn.
* **Claude → OpenAI**: build SSE chunks and the non-streaming response body, map
  ``stop_reason`` → ``finish_reason``, and derive ``usage`` from the ``result`` line.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any, Optional

from app.events import TurnDone
from app.openai_models import ChatMessage

# ── ids / timestamps ─────────────────────────────────────────────────────────


def new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def now() -> int:
    return int(time.time())


# ── OpenAI message helpers ─────────────────────────────────────────────────—


def message_text(msg: ChatMessage) -> str:
    """Flatten an OpenAI message's ``content`` to plain text.

    Handles the string form and the content-parts list form
    (``[{"type":"text","text":...}, ...]``); non-text parts (images) are skipped.
    """
    c = msg.content
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    parts: list[str] = []
    for part in c:
        if isinstance(part, dict):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)


_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_MAX_IMAGE_BASE64_BYTES = 20 * 1024 * 1024


def _claude_message_content(msg: ChatMessage) -> str | list[dict[str, Any]]:
    """Preserve OpenAI data-URL images as Claude stream-json image blocks."""
    if not isinstance(msg.content, list):
        return message_text(msg)

    blocks: list[dict[str, Any]] = []
    has_image = False
    for part in msg.content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            # Skip empties: the API rejects {"type":"text","text":""}, so an
            # empty text part sitting next to an image would fail the whole
            # turn. Dropping it matches the string path, which contributes
            # nothing for an empty part either.
            if part["text"]:
                blocks.append({"type": "text", "text": part["text"]})
            continue
        if part.get("type") != "image_url":
            continue

        image_url = part.get("image_url")
        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
        header, sep, data = str(url).partition(",")
        media_type = header[5:].split(";", 1)[0].lower() if header.startswith("data:") else ""
        if (
            not sep
            or ";base64" not in header.lower()
            or media_type not in _IMAGE_MEDIA_TYPES
            or not data
            or len(data) > _MAX_IMAGE_BASE64_BYTES
        ):
            continue
        try:
            base64.b64decode(data, validate=True)
        except ValueError:
            continue
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })
        has_image = True

    return blocks if has_image else message_text(msg)


def split_system(messages: list[ChatMessage]) -> tuple[list[ChatMessage], Optional[str]]:
    """Partition into (non-system messages, joined system prompt or None)."""
    system_parts: list[str] = []
    convo: list[ChatMessage] = []
    for m in messages:
        if m.role == "system":
            t = message_text(m)
            if t:
                system_parts.append(t)
        else:
            convo.append(m)
    system = "\n\n".join(system_parts) if system_parts else None
    return convo, system


def _role_label(role: str) -> str:
    return {"user": "User", "assistant": "Assistant", "tool": "Tool"}.get(role, role.capitalize())


def _history_images(msg: ChatMessage) -> list[dict[str, Any]]:
    """Image blocks carried by a message, or ``[]``.

    Goes through :func:`_claude_message_content` so an image folded from history
    is validated exactly like one on the live turn (data-URL shape, media type,
    size cap, base64).
    """
    content = _claude_message_content(msg)
    if not isinstance(content, list):
        return []
    return [b for b in content if b.get("type") == "image"]


def _render_items(items: list[Any]) -> str | list[dict[str, Any]]:
    """Render fold items — labelled lines and image blocks — for the wire.

    An all-text fold collapses back to one newline-joined string, the exact
    shape this server has always sent, so a conversation without images is
    byte-identical to before. As soon as an image is in play the fold becomes a
    content-block list instead, with each run of lines merged into a single text
    block so every image keeps its position in the transcript.
    """
    if not any(isinstance(it, dict) for it in items):
        return "\n".join(items)

    blocks: list[dict[str, Any]] = []
    run: list[str] = []
    for it in items:
        if isinstance(it, dict):
            if run:
                blocks.append({"type": "text", "text": "\n".join(run)})
                run = []
            blocks.append(it)
        else:
            run.append(it)
    if run:
        blocks.append({"type": "text", "text": "\n".join(run)})
    return blocks


def fold_conversation(convo: list[ChatMessage]) -> str | list[dict[str, Any]]:
    """Fold a (system-stripped) OpenAI conversation into one Claude user turn.

    A lone trailing user message is sent as-is. Otherwise prior turns become a
    transcript preamble so a fresh, stateless subprocess still has the context.

    Images in those prior turns are carried through as content blocks in place,
    so a refold does not blind the model to a picture it was already shown — a
    follow-up like "what colour was the shirt?" used to be answered from the
    text alone, confidently and with nothing in the logs. A conversation
    without images folds to exactly the same string as before.

    This path matters more than it looks: every MODE_LEGACY turn folds, and the
    content lane now goes legacy for the rest of a conversation after a restart
    (see app.session_reuse).
    """
    if not convo:
        return ""
    if len(convo) == 1 and convo[0].role == "user":
        return _claude_message_content(convo[0])

    items: list[Any] = []
    for m in convo[:-1]:
        text = message_text(m)
        if m.tool_calls:
            calls = ", ".join(
                f"{tc.function.name}({tc.function.arguments or ''})" for tc in m.tool_calls
            )
            text = (text + " " if text else "") + f"[called tools: {calls}]"
        images = _history_images(m)
        if text:
            items.append(f"{_role_label(m.role)}: {text}")
        elif images:
            # An image-only turn still needs its label, so the blocks that
            # follow are attributed to the right speaker.
            items.append(f"{_role_label(m.role)}:")
        items.extend(images)

    last = convo[-1]
    last_content = _claude_message_content(last)

    if isinstance(last_content, list):
        if items:
            preamble = _render_items(
                ["Conversation so far:", *items, "", f"{_role_label(last.role)}:"]
            )
            if isinstance(preamble, str):
                return [{"type": "text", "text": preamble}, *last_content]
            return [*preamble, *last_content]
        return last_content

    last_line = f"{_role_label(last.role)}: {last_content}"
    if items:
        # The trailing "" reproduces the historical "\n\n" seam before the live
        # turn, so a text-only fold is byte-identical to the previous output.
        return _render_items(["Conversation so far:", *items, "", last_line])
    return message_text(last)


def turn_delta(convo: list[ChatMessage]) -> str | list[dict[str, Any]]:
    """The trailing user turn's content only — for a ``--resume`` spawn that
    already holds the prior context on disk (Phase 3).

    Preserves multimodal (image) content via :func:`_claude_message_content`.
    Falls back to :func:`fold_conversation` when the trailing message is not a
    plain user message; on a fresh (non-continuation) turn the last message is
    always the new user turn, so the fast path is the norm.
    """
    if convo and convo[-1].role == "user":
        return _claude_message_content(convo[-1])
    return fold_conversation(convo)


# ── stop_reason / usage mapping ────────────────────────────────────────────—


def map_finish_reason(stop_reason: Optional[str]) -> str:
    """Claude ``stop_reason`` → OpenAI ``finish_reason``."""
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason or "", "stop")


def usage_from_turn(turn: TurnDone) -> dict[str, Any]:
    """Derive an OpenAI ``usage`` object from a ``result`` line.

    The top-level ``usage`` on the ``result`` line reflects the *last* model
    iteration only; when the CLI reports a per-iteration breakdown we sum it so
    internal (built-in) tool turns are counted. Cache tokens count as prompt
    tokens. Never fabricates — unknown values stay 0. Surfaces ``total_cost_usd``
    as the non-standard ``cost_usd``.
    """
    u = turn.usage or {}
    iterations = u.get("iterations")
    if isinstance(iterations, list) and iterations:
        prompt = 0
        completion = 0
        for it in iterations:
            if not isinstance(it, dict):
                continue
            prompt += _int(it.get("input_tokens"))
            prompt += _int(it.get("cache_read_input_tokens"))
            prompt += _int(it.get("cache_creation_input_tokens"))
            completion += _int(it.get("output_tokens"))
    else:
        prompt = (
            _int(u.get("input_tokens"))
            + _int(u.get("cache_read_input_tokens"))
            + _int(u.get("cache_creation_input_tokens"))
        )
        completion = _int(u.get("output_tokens"))
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if turn.total_cost_usd is not None:
        usage["cost_usd"] = turn.total_cost_usd
    return usage


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


# ── SSE chunk builders ─────────────────────────────────────────────────────—

DONE = "data: [DONE]\n\n"


def sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _chunk(
    cid: str,
    model: str,
    created: int,
    *,
    delta: dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


def role_chunk(cid: str, model: str, created: int) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"role": "assistant", "content": ""})


def text_chunk(cid: str, model: str, created: int, text: str) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"content": text})


def tool_calls_chunk(
    cid: str, model: str, created: int, tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={"tool_calls": tool_calls})


def finish_chunk(
    cid: str,
    model: str,
    created: int,
    finish_reason: str,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _chunk(cid, model, created, delta={}, finish_reason=finish_reason, usage=usage)


# ── non-streaming response ─────────────────────────────────────────────────—


def completion_response(
    cid: str,
    model: str,
    created: int,
    *,
    content: Optional[str],
    finish_reason: str,
    usage: Optional[dict[str, Any]] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    obj: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj
