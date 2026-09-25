"""The autonomous (tool-less) route must keep out of the content reuse lane.

hermes' title generator is a tool-less side call with no session id whose only
message is the chat's opening line. On 2026-09-25 it seeded a content-lane
session under the title prompt, a later tooled chat turn that arrived without
an hsid resumed into it, and the TUI showed {"title": ...} as the reply. The
registry-level tests cover `plan(has_tools=False)`; this one guards the route
actually passing it.
"""

import httpx

from app import main as main_mod
from app.config import Settings
from app.conversation import ConversationManager
from app.mcp_bridge import McpBridge
from app.session_reuse import SessionRegistry
from tests.test_concurrency_cap_autonomous import _StubSession


async def test_autonomous_route_skips_content_lane(monkeypatch):
    import app.routes.chat as chatmod

    s = Settings(host="127.0.0.1", api_key=None, session_reuse=True)
    monkeypatch.setattr(main_mod, "get_settings", lambda: s)
    monkeypatch.setattr(chatmod, "get_settings", lambda: s)
    monkeypatch.setattr(chatmod, "ClaudeSession", _StubSession)
    app = main_mod.create_app()
    app.state.conv_manager = ConversationManager(McpBridge(), s)
    reg = SessionRegistry()
    app.state.session_registry = reg

    seen = []
    orig = reg.plan

    def spy(*a, **kw):
        plan = orig(*a, **kw)
        seen.append((kw.get("has_tools", True), plan.mode))
        return plan

    monkeypatch.setattr(reg, "plan", spy)

    body = {"model": "claude-opus-5",
            "messages": [{"role": "user", "content": "update complete"}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as c:
        await c.post("/v1/chat/completions", json=body)

    assert seen == [(False, "legacy")]
    assert not reg._sessions  # nothing seeded for a tooled turn to fall into
