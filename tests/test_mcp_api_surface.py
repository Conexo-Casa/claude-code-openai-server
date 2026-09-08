"""Pin the `mcp` API surface this server is actually built on.

`app/mcp_bridge.py` uses the low-level decorator API — `@server.list_tools()`
and `@server.call_tool(validate_input=False)` — which mcp 2.0 removes. The
dependency is bounded `mcp>=1.2,<2` in pyproject.toml, but a bound alone fails
silently in the wrong direction: if someone widens it, or installs into an
existing venv out-of-band, the breakage shows up as an ImportError at process
start. For a systemd unit that means the service just stops coming back, with
the real cause buried in the journal.

These tests make that a loud test failure instead. They are guards, not a
reproduction of a current bug — nothing here fails on mcp 1.x.
"""

from __future__ import annotations

import importlib.metadata as md
import inspect

from mcp.server.lowlevel import Server


def test_installed_mcp_is_below_2():
    version = md.version("mcp")
    major = int(version.split(".")[0])
    assert major < 2, (
        f"mcp {version} is installed, but app/mcp_bridge.py is built on the 1.x "
        "low-level decorator API. Port the bridge before widening the bound."
    )


def test_lowlevel_decorator_api_exists():
    """The two decorators mcp_bridge registers its handlers with."""
    server = Server("surface-probe")
    for name in ("list_tools", "call_tool"):
        attr = getattr(server, name, None)
        assert callable(attr), f"Server.{name} is missing — mcp API changed"


def test_call_tool_still_accepts_validate_input():
    """We pass validate_input=False; losing that kwarg is a silent behavior change.

    Input validation belongs to the OpenAI client's schema, not ours — the
    bridge forwards arguments verbatim.
    """
    server = Server("surface-probe")
    params = inspect.signature(server.call_tool).parameters
    assert "validate_input" in params, (
        "Server.call_tool no longer accepts validate_input — mcp_bridge would "
        "start validating tool arguments against its own schema"
    )


def test_streamable_http_session_manager_is_importable():
    """The transport the /mcp mount is built on."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    params = inspect.signature(StreamableHTTPSessionManager).parameters
    for kwarg in ("app", "json_response", "stateless"):
        assert kwarg in params, f"StreamableHTTPSessionManager lost {kwarg!r}"


def test_mcp_types_module_is_importable():
    import mcp.types as mtypes

    # The concrete types mcp_bridge constructs in its handlers.
    for name in ("Tool", "TextContent"):
        assert hasattr(mtypes, name), f"mcp.types.{name} is missing"
