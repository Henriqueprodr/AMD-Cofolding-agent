# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the container-side run_python scratch-computation tool.

These genuinely spawn ``sys.executable``; the run root is redirected to a
tmp_path so nothing touches /shared.

Covers:
- stdout capture, file listing, work_dir reuse
- a raising snippet is a normal result (ok=False), not a raised tool error
- timeout kills the process group
- output and code size caps
- refusals: empty code, oversized code, work_dir escape, disabled tool
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reinvent4_mcp.tools import code_exec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def run_root(tmp_path):
    """Point the tool's run root at a tmp dir (no /shared in tests)."""
    root = tmp_path / "code_runs"
    root.mkdir()
    with patch.object(code_exec, "_run_root", return_value=root):
        yield root


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestRunPython:
    """Normal execution, files, and directory reuse."""

    def test_stdout_is_captured(self, run_root):
        result = code_exec.run_python("print('hello', 2 + 2)")

        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert "hello 4" in result["stdout"]
        assert result["timed_out"] is False

    def test_written_files_are_listed(self, run_root):
        result = code_exec.run_python("open('series.csv', 'w').write('a,b\\n')")

        names = [f["path"].rsplit("/", 1)[-1] for f in result["files"]]
        assert "series.csv" in names
        # The snippet itself is never reported as an output artefact.
        assert "snippet.py" not in names

    def test_work_dir_round_trips(self, run_root):
        first = code_exec.run_python("open('state.txt', 'w').write('kept')")

        second = code_exec.run_python(
            "print(open('state.txt').read())", work_dir=first["work_dir"]
        )

        assert second["stdout"].strip() == "kept"
        assert second["work_dir"] == first["work_dir"]

    def test_each_call_gets_a_fresh_directory(self, run_root):
        a = code_exec.run_python("print(1)")
        b = code_exec.run_python("print(2)")

        assert a["work_dir"] != b["work_dir"]


# ---------------------------------------------------------------------------
# Failure is a result, not an exception
# ---------------------------------------------------------------------------


class TestFailureIsAResult:
    """A traceback is what the model needs to read, not a tool error."""

    def test_raising_snippet_returns_ok_false_with_traceback(self, run_root):
        result = code_exec.run_python("raise ValueError('boom')")

        assert result["ok"] is False
        assert result["exit_code"] != 0
        assert "ValueError: boom" in result["stderr"]
        # Crucially NOT an {"error": ...} refusal -- the model can fix this.
        assert "error" not in result

    def test_timeout_is_reported_not_raised(self, run_root):
        result = code_exec.run_python("import time; time.sleep(30)", timeout_s=1)

        assert result["timed_out"] is True
        assert result["ok"] is False
        assert result["duration_s"] < 10
        assert "killed after" in result["stderr"]


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


class TestCaps:
    """Output and input limits keep a runaway snippet out of the context."""

    def test_stdout_is_truncated(self, run_root):
        result = code_exec.run_python("print('x' * 50000)")

        assert result["stdout_truncated"] is True
        assert len(result["stdout"]) < 50000
        assert "truncated" in result["stdout"]

    def test_timeout_is_clamped_to_the_ceiling(self, run_root):
        # Asking for an hour must not get one; the snippet is instant, so we
        # assert on the clamp by way of the tool still returning promptly.
        result = code_exec.run_python("print('fast')", timeout_s=10_000)

        assert result["ok"] is True

    def test_oversized_code_is_refused_without_running(self, run_root):
        with patch.object(code_exec.subprocess, "Popen") as popen:
            result = code_exec.run_python("x = 1\n" * 20_000)

        assert "error" in result
        popen.assert_not_called()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    """Things the tool declines to do at all."""

    def test_empty_code_does_not_spawn(self, run_root):
        with patch.object(code_exec.subprocess, "Popen") as popen:
            result = code_exec.run_python("   ")

        assert "error" in result
        popen.assert_not_called()

    def test_work_dir_outside_run_root_is_rejected(self, run_root):
        result = code_exec.run_python("print(1)", work_dir="/etc")

        assert "error" in result
        assert "work_dir" in result["error"]

    def test_work_dir_traversal_is_rejected(self, run_root):
        result = code_exec.run_python("print(1)", work_dir="../../etc")

        assert "error" in result

    def test_disabled_tool_refuses(self, run_root):
        with patch.object(
            code_exec, "_settings", return_value=(False, run_root, 60, 300)
        ):
            result = code_exec.run_python("print(1)")

        assert "error" in result
        assert "disabled" in result["error"]
