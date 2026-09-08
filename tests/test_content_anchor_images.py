"""The content-lane anchor must see images, not just text.

`content_anchor` keyed on `message_text(convo[0])`, and `message_text` drops
non-text content parts by design. So two unrelated conversations that opened
with the same sentence but *different* attached images produced a byte-identical
anchor, hence the same `uuid5` session UUID — and the content lane would
`--resume` one conversation into the other's history, handing user B the image
and transcript of user A.

Two properties are asserted together, and they pull against each other:

* **Discrimination** — differing non-text parts must yield differing anchors.
* **Backward compatibility** — for string content, and for a parts list that is
  entirely text, the anchor must be *unchanged*. The anchor is the key for
  sessions already on disk; changing it for existing traffic would silently
  re-seed every live conversation.
"""

from __future__ import annotations

import json

from app.openai_models import ChatMessage
from app.session_reuse import _normalize, content_anchor, derive_content_session_uuid


def user(content) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def img(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


TEXT = {"type": "text", "text": "describe this"}


# ── the bug ────────────────────────────────────────────────────────────────

def test_same_text_different_images_get_different_anchors():
    a = content_anchor([user([TEXT, img("data:image/png;base64,AAAA")])])
    b = content_anchor([user([TEXT, img("data:image/png;base64,BBBB")])])
    assert a is not None and b is not None
    assert a != b, "identical anchor for different images — sessions would merge"


def test_same_text_different_images_get_different_session_uuids():
    """The anchor feeds uuid5; equal anchors mean one shared session."""
    a = content_anchor([user([TEXT, img("data:image/png;base64,AAAA")])])
    b = content_anchor([user([TEXT, img("data:image/png;base64,BBBB")])])
    assert derive_content_session_uuid(a) != derive_content_session_uuid(b)


def test_identical_content_still_matches():
    """Discrimination must not cost stability — same input, same key."""
    msg = [TEXT, img("data:image/png;base64,AAAA")]
    assert content_anchor([user(list(msg))]) == content_anchor([user(list(msg))])


def test_anchor_ignores_dict_key_ordering():
    """json.dumps(sort_keys=True): part key order must not change the key."""
    a = content_anchor([user([{"type": "image_url", "image_url": {"url": "u"}}])])
    b = content_anchor([user([{"image_url": {"url": "u"}, "type": "image_url"}])])
    assert a == b


# ── backward compatibility: existing sessions must keep their key ──────────

def test_string_content_anchor_is_unchanged():
    text = "  hello world  "
    assert content_anchor([user(text)]) == _normalize(text)


def test_text_only_parts_list_anchor_is_unchanged():
    """A text-only list must key identically to the concatenated text.

    This is what the old message_text() path produced, so on-disk sessions for
    text-only conversations keep working.
    """
    parts = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert content_anchor([user(parts)]) == "hello world"


def test_bare_string_parts_are_still_treated_as_text():
    assert content_anchor([user(["hello ", "world"])]) == "hello world"


def test_no_nontext_marker_when_content_is_text_only():
    """The \\x1f marker must be absent for text-only content, or the key moved."""
    assert "\x1f" not in (content_anchor([user("plain")]) or "")
    assert "\x1f" not in (content_anchor([user([TEXT])]) or "")


# ── unusable anchors still fall back ───────────────────────────────────────

def test_non_user_leading_message_has_no_anchor():
    assert content_anchor([ChatMessage(role="assistant", content="hi")]) is None


def test_empty_conversation_has_no_anchor():
    assert content_anchor([]) is None


def test_null_content_has_no_anchor():
    assert content_anchor([user(None)]) is None


def test_image_only_opening_now_gets_an_anchor():
    """Deliberate behaviour change.

    Previously an image-only first message flattened to "" and fell back to
    legacy. It now anchors on the image digest — so it gets reuse, and two
    different image-only openings no longer collide.
    """
    a = content_anchor([user([img("data:image/png;base64,AAAA")])])
    b = content_anchor([user([img("data:image/png;base64,BBBB")])])
    assert a is not None and b is not None
    assert a != b


# ── the digest is bounded, not the raw image ───────────────────────────────

def test_large_image_does_not_bloat_the_anchor():
    """A data URL can be megabytes; the anchor must stay small (runs per turn)."""
    huge = "data:image/png;base64," + ("A" * 2_000_000)
    anchor = content_anchor([user([TEXT, img(huge)])])
    assert anchor is not None
    assert len(anchor) < 200, f"anchor is {len(anchor)} chars — image embedded raw?"
    assert huge not in anchor


def test_text_part_cannot_forge_the_nontext_section():
    """A crafted text part must not be able to spoof another anchor's digest."""
    real = content_anchor([user([TEXT, img("data:image/png;base64,AAAA")])])
    assert real is not None
    forged = content_anchor([user([{"type": "text", "text": real}])])
    assert forged != real
