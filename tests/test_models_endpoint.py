"""``/v1/models`` must advertise exactly what ``resolve_model`` will accept.

The endpoint used to keep its own hardcoded id list, independent of
``config.KNOWN_MODEL_ALIASES``. The two drifted: ``/v1/models`` advertised a
superseded ``claude-sonnet-4-6``, omitted the ``opusplan``/``default`` aliases
that ``resolve_model`` accepts, and never listed the concrete model the server
actually defaults to. These tests pin the endpoint to the shared source of
truth so a future model rename can't silently desync it again.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.routes.models as models_mod
from app import main as main_mod
from app.config import KNOWN_MODEL_ALIASES, Settings, resolve_model


def _advertised(monkeypatch, default_model: str) -> tuple[list[str], Settings]:
    """Return the ids served by ``GET /v1/models`` for a given default model.

    Init kwargs outrank environment variables in pydantic-settings, so the
    fields set here are hermetic even when the ambient shell carries the
    service's real ``CCI_*`` config.
    """
    settings = Settings(host="127.0.0.1", api_key=None, default_model=default_model)
    # raising=False: before the derived-list fix, routes.models had no
    # get_settings reference at all, so this patch is a no-op there and the
    # assertions below fail on the stale static list rather than erroring.
    monkeypatch.setattr(models_mod, "get_settings", lambda: settings, raising=False)
    monkeypatch.setattr(main_mod, "get_settings", lambda: settings)

    response = TestClient(main_mod.create_app()).get("/v1/models")
    assert response.status_code == 200
    return [card["id"] for card in response.json()["data"]], settings


def test_every_advertised_id_is_accepted_by_resolve_model(monkeypatch):
    ids, settings = _advertised(monkeypatch, "claude-opus-5")
    for mid in ids:
        assert resolve_model(mid, settings) == mid, (
            f"{mid!r} is advertised but resolve_model rewrites it to the default"
        )


def test_every_accepted_alias_is_advertised(monkeypatch):
    ids, _ = _advertised(monkeypatch, "claude-opus-5")
    missing = KNOWN_MODEL_ALIASES - set(ids)
    assert not missing, f"resolve_model accepts these but /v1/models hides them: {missing}"


def test_configured_default_is_advertised_first(monkeypatch):
    ids, _ = _advertised(monkeypatch, "claude-opus-5")
    assert ids[0] == "claude-opus-5", (
        "clients that pick data[0] must get the model this server actually uses"
    )


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8"])
def test_advertised_default_tracks_config(monkeypatch, model):
    """Changing CCI_DEFAULT_MODEL must not require editing routes/models.py."""
    ids, _ = _advertised(monkeypatch, model)
    assert model in ids


def test_no_unreachable_dated_ids_are_advertised(monkeypatch):
    """Only the configured default may be a concrete id; the rest are aliases.

    A hardcoded dated id is precisely what goes stale — it survives in this list
    long after the model behind it is superseded, and nothing fails loudly.
    """
    ids, _ = _advertised(monkeypatch, "claude-opus-5")
    concrete = [mid for mid in ids if mid not in KNOWN_MODEL_ALIASES]
    assert concrete == ["claude-opus-5"], (
        f"unexpected hardcoded concrete ids in /v1/models: {concrete}"
    )


def test_ordering_is_stable_across_calls(monkeypatch):
    """KNOWN_MODEL_ALIASES is a set; string hashing is randomised per process."""
    first, _ = _advertised(monkeypatch, "claude-opus-5")
    second, _ = _advertised(monkeypatch, "claude-opus-5")
    assert first == second
