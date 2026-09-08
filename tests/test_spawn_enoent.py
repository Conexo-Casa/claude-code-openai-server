"""A failed spawn must name the thing that was actually missing.

`create_subprocess_exec` raises FileNotFoundError for two different causes: the
binary is not on PATH, or `cwd` does not exist. The handler blamed PATH
unconditionally, which sends you hunting for an installation problem when the
workdir is what is wrong — upstream's commit message notes this is exactly what
made an unrelated bug look like an environment fault.

CPython records the path that actually failed in `.filename`, so both branches
are distinguishable.
"""

from __future__ import annotations

import pytest

from app.claude_session import ClaudeSession


def _session(**kw) -> ClaudeSession:
    base = dict(
        claude_bin="claude",
        model="claude-opus-5",
        permission_mode="bypassPermissions",
        workdir="/tmp",
    )
    base.update(kw)
    return ClaudeSession(**base)


async def test_missing_binary_blames_the_binary(monkeypatch):
    sess = _session(claude_bin="/nonexistent/claude-binary")

    async def boom(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory",
                                "/nonexistent/claude-binary")

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    with pytest.raises(RuntimeError) as ei:
        await sess.start()
    msg = str(ei.value)
    assert "claude CLI" in msg and "/nonexistent/claude-binary" in msg
    assert "workdir" not in msg, f"blamed the workdir for a missing binary: {msg}"


async def test_missing_workdir_blames_the_workdir(monkeypatch):
    sess = _session(workdir="/nonexistent/workdir")

    async def boom(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory",
                                "/nonexistent/workdir")

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    with pytest.raises(RuntimeError) as ei:
        await sess.start()
    msg = str(ei.value)
    assert "workdir" in msg and "/nonexistent/workdir" in msg
    assert "not found on PATH" not in msg, (
        f"blamed PATH for a missing workdir: {msg}"
    )


async def test_enoent_without_a_filename_still_reports_something(monkeypatch):
    """Defensive: .filename can be None. Must not render 'None'."""
    sess = _session(claude_bin="claude")

    async def boom(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    with pytest.raises(RuntimeError) as ei:
        await sess.start()
    msg = str(ei.value)
    assert "cannot spawn claude" in msg
    assert "'claude'" in msg, f"lost the binary name: {msg}"


async def test_error_is_chained_for_the_traceback(monkeypatch):
    sess = _session()

    async def boom(*_a, **_k):
        raise FileNotFoundError(2, "No such file or directory", "claude")

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    with pytest.raises(RuntimeError) as ei:
        await sess.start()
    assert isinstance(ei.value.__cause__, FileNotFoundError)
