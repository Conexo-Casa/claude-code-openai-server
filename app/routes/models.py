"""``GET /v1/models`` — advertise the Claude models this server exposes.

The list is derived, not hand-maintained. Aliases come from
``config.KNOWN_MODEL_ALIASES`` — the same set ``config.resolve_model`` gates on —
followed by the configured ``default_model``. Everything advertised here is
therefore something ``resolve_model`` will actually pass through to the CLI, and
the concrete default appears first so a client that naively picks ``data[0]``
gets the model this server really uses.

Deliberately no dated concrete ids (``claude-opus-4-8`` and friends). As
``config.KNOWN_MODEL_ALIASES`` notes, the CLI resolves an alias to the current
concrete id on its own, so pinning them here only creates a second list to keep
in sync — which is exactly how this endpoint came to advertise a superseded
Sonnet while omitting the model the server defaults to.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import KNOWN_MODEL_ALIASES, get_settings
from app.openai_models import ModelCard, ModelList

router = APIRouter()


def _model_ids() -> list[str]:
    """Concrete default first, then every alias the CLI accepts, sorted.

    Sorted rather than set-ordered: ``KNOWN_MODEL_ALIASES`` is a ``set`` of
    strings, and CPython randomises string hashing per process, so iterating it
    directly would reorder this endpoint's output on every restart.
    """
    default = get_settings().default_model.strip()
    ids = [default] if default else []
    ids += [alias for alias in sorted(KNOWN_MODEL_ALIASES) if alias != default]
    return ids


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=mid) for mid in _model_ids()])


@router.get("/v1/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str) -> ModelCard:
    return ModelCard(id=model_id)
