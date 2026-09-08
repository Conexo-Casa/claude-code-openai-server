"""A refold must not blind the model to a picture it was already shown.

`fold_conversation` walked prior turns through `message_text`, which drops
non-text parts. So an image sent on turn 1 vanished when turn 2 refolded the
conversation, and a follow-up like "what colour was the shirt?" was answered
from the surrounding text alone — confidently, with nothing in the logs to say
the image was missing. Only the live turn kept its images.

This is the same `message_text` blindness fixed in the session-reuse anchor
(42fb490), in a second place.

It also matters more than it first appears: every MODE_LEGACY turn folds, and
61bce14 made the content lane go legacy for the rest of a conversation after a
restart — so this path carries more traffic now, not less.

The hard constraint, as with the anchor: a conversation with no images must fold
to the byte-identical string as before. The existing tests in test_translate.py
assert exact fold output and are the real guard on that; these add the image
cases.
"""

from __future__ import annotations

import base64

from app.openai_models import ChatMessage
from app.translate import fold_conversation

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode()
DATA_URL = f"data:image/png;base64,{PNG}"


def u(content) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def a(content) -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


def img(url: str = DATA_URL) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def txt(s: str) -> dict:
    return {"type": "text", "text": s}


def _images(folded) -> list[dict]:
    if not isinstance(folded, list):
        return []
    return [b for b in folded if b.get("type") == "image"]


# ── the bug ────────────────────────────────────────────────────────────────

def test_history_image_survives_the_refold():
    convo = [
        u([txt("what is this?"), img()]),
        a("a small png"),
        u("what colour was it?"),
    ]
    folded = fold_conversation(convo)
    assert isinstance(folded, list), "fold collapsed to text and lost the image"
    assert len(_images(folded)) == 1, "the history image did not survive"


def test_follow_up_still_sees_the_image_many_turns_later():
    convo = [u([txt("look"), img()]), a("ok")]
    for i in range(6):
        convo.append(u(f"question {i}"))
        convo.append(a(f"answer {i}"))
    convo.append(u("and the first picture?"))
    assert len(_images(fold_conversation(convo))) == 1


def test_multiple_history_images_all_survive():
    convo = [
        u([txt("first"), img()]),
        a("ok"),
        u([txt("second"), img()]),
        a("ok"),
        u("compare them"),
    ]
    assert len(_images(fold_conversation(convo))) == 2


# ── ordering: an image must stay where it was said ─────────────────────────

def test_image_keeps_its_position_in_the_transcript():
    convo = [
        u("before the picture"),
        u([txt("here it is"), img()]),
        u("after the picture"),
    ]
    folded = fold_conversation(convo)
    assert isinstance(folded, list)
    kinds = [b["type"] for b in folded]
    i = kinds.index("image")
    before = "\n".join(b.get("text", "") for b in folded[:i])
    after = "\n".join(b.get("text", "") for b in folded[i + 1:])
    assert "before the picture" in before
    assert "after the picture" in after


def test_image_only_history_turn_still_gets_a_speaker_label():
    """Otherwise the blocks that follow are attributed to the wrong speaker."""
    convo = [u([img()]), a("I see it"), u("describe it again")]
    folded = fold_conversation(convo)
    assert isinstance(folded, list)
    text = "\n".join(b.get("text", "") for b in folded if b["type"] == "text")
    assert "User:" in text


# ── byte-compatibility for the no-image case ───────────────────────────────

def test_text_only_fold_is_still_a_plain_string():
    convo = [u("hello"), a("hi"), u("more")]
    folded = fold_conversation(convo)
    assert isinstance(folded, str), "a text-only fold must not become blocks"


def test_text_only_fold_keeps_its_exact_shape():
    convo = [u("hello"), a("hi"), u("more")]
    assert fold_conversation(convo) == (
        "Conversation so far:\nUser: hello\nAssistant: hi\n\nUser: more"
    )


def test_single_user_message_unchanged():
    assert fold_conversation([u("just this")]) == "just this"


# ── empty text blocks are rejected by the API ──────────────────────────────

def test_empty_text_part_beside_an_image_is_dropped():
    """{"type":"text","text":""} fails the whole turn if it reaches the wire."""
    folded = fold_conversation([u([txt(""), img()])])
    assert isinstance(folded, list)
    assert all(b.get("text") != "" for b in folded if b["type"] == "text")
    assert len(_images(folded)) == 1


def test_no_empty_text_blocks_anywhere_in_a_folded_result():
    convo = [
        u([txt(""), img()]),
        a(""),
        u([txt(""), img()]),
        u("go"),
    ]
    folded = fold_conversation(convo)
    if isinstance(folded, list):
        assert all(b.get("text") != "" for b in folded if b["type"] == "text")


# ── live-turn images keep working ──────────────────────────────────────────

def test_live_turn_image_still_preserved():
    convo = [u("context line"), a("ok"), u([txt("and this?"), img()])]
    assert len(_images(fold_conversation(convo))) == 1


def test_history_and_live_images_both_present():
    convo = [u([txt("first"), img()]), a("ok"), u([txt("second"), img()])]
    assert len(_images(fold_conversation(convo))) == 2


# ── malformed images are dropped, not carried ──────────────────────────────

def test_invalid_history_image_is_not_carried():
    convo = [u([txt("look"), img("data:image/png;base64,!!!not-base64!!!")]),
             a("ok"), u("again?")]
    folded = fold_conversation(convo)
    assert _images(folded) == []
    assert isinstance(folded, str), "an unusable image should leave a text fold"
