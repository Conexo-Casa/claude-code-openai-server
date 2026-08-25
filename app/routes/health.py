"""Health/info endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app import __version__

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    body: dict = {
        "status": "ok",
        "service": "claude-code-interface",
        "version": __version__,
    }
    # Absent until the lifespan builds the manager; report what we have rather
    # than 500-ing a health check during startup.
    mgr = getattr(request.app.state, "conv_manager", None)
    if mgr is not None:
        body["conversations"] = mgr.lane_usage()
    return body


@router.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "claude-code-interface",
        "version": __version__,
        "openai_base": "/v1",
    }
