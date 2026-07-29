"""Tests for the developer assistant.

The load-bearing tests here are the ones proving the operation catalogue is a
catalogue and not a shell: that an arbitrary command cannot be run, that a commit
message cannot become a second command, and that work is confined to permitted
directories.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quainex.config.settings import Settings
from quainex.core.devtools import DevRunner, operation_catalogue, resolve_operation
from quainex.core.devtools.assistant import CodeAssistant
from quainex.core.devtools.operations import MAX_MESSAGE_CHARS
from quainex.core.devtools.runner import MAX_OUTPUT_CHARS, _tail
from quainex.core.exceptions import CommandExecutionError, CommandNotAllowedError
from tests.test_brain import FakeProvider


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "t.db",
        command_search_roots=[tmp_path],
    )


# -- the catalogue is a catalogue, not a shell ----------------------------


@pytest.mark.parametrize("key", ["git.status", "git.commit", "tests.run", "docker.ps", "lint.run"])
def test_known_operations_resolve(key):
    assert resolve_operation(key) is not None


@pytest.mark.parametrize(
    "key",
    [
        "",
        "rm -rf /",
        "git",
        "git.push --force",
        "git status; rm -rf /",
        "python",
        "bash",
        "curl",
        "git.rebase",
        "docker.run",
    ],
)
def test_anything_outside_the_catalogue_is_refused(key):
    # Note "git" alone does not resolve. Allowing an executable would allow
    # everything that executable can be talked into — `git -c core.pager=...`
    # reaches arbitrary execution without ever leaving "git".
    assert resolve_operation(key) is None


def test_resolution_is_exact_not_fuzzy():
    # "git.push" and "git.pull" differ by one character and do very different
    # things. A near-miss here would be acted on.
    assert resolve_operation("git.pus") is None
    assert resolve_operation("git.puush") is None


def test_catalogue_lists_operations():
    catalogue = operation_catalogue()
    assert "git.status" in catalogue
    assert all(isinstance(summary, str) and summary for summary in catalogue.values())


def test_mutating_operations_are_marked():
    for key in ("git.add", "git.commit", "git.push", "git.pull"):
        operation = resolve_operation(key)
        assert operation is not None
        assert operation.mutating is True, f"{key} changes state and must be flagged"

    for key in ("git.status", "git.log", "tests.run", "docker.ps"):
        operation = resolve_operation(key)
        assert operation is not None
        assert operation.mutating is False


# -- argv construction -----------------------------------------------------


def test_unknown_operation_is_refused_with_alternatives(tmp_path):
    runner = DevRunner(_settings(tmp_path))
    with pytest.raises(CommandNotAllowedError, match="not an available operation"):
        runner.run("definitely.not.real")


def test_commit_without_a_message_is_refused(tmp_path):
    runner = DevRunner(_settings(tmp_path))
    with pytest.raises(CommandNotAllowedError, match="needs a message"):
        runner.run("git.commit")


def test_oversized_commit_message_is_refused(tmp_path):
    runner = DevRunner(_settings(tmp_path))
    with pytest.raises(CommandNotAllowedError, match="limit is"):
        runner.run("git.commit", message="x" * (MAX_MESSAGE_CHARS + 1))


def test_shell_metacharacters_in_a_message_stay_one_argument(tmp_path):
    # There is no shell, so a message full of operators is still one argv
    # element. This asserts the argv itself rather than trusting that claim.
    runner = DevRunner(_settings(tmp_path))
    hostile = 'fix stuff" && rm -rf / #'
    argv = runner._build_argv(("git", "commit", "-m", "{message}"), True, hostile)

    assert argv == ["git", "commit", "-m", hostile]
    assert len(argv) == 4, "the message must not split into extra arguments"


def test_null_bytes_in_a_message_are_refused(tmp_path):
    runner = DevRunner(_settings(tmp_path))
    with pytest.raises(CommandNotAllowedError, match="not allowed"):
        runner._build_argv(("git", "commit", "-m", "{message}"), True, "a\x00b")


# -- directory containment -------------------------------------------------


def test_directory_defaults_to_the_first_permitted_root(tmp_path):
    runner = DevRunner(_settings(tmp_path))
    assert runner._resolve_directory(None) == tmp_path.resolve()


def test_directory_inside_the_root_is_accepted(tmp_path):
    (tmp_path / "project").mkdir()
    runner = DevRunner(_settings(tmp_path))
    assert runner._resolve_directory("project") == (tmp_path / "project").resolve()


@pytest.mark.parametrize("escape", ["..", "../..", "C:\\Windows", "../../../Windows"])
def test_directories_outside_the_roots_are_refused(tmp_path, escape):
    runner = DevRunner(_settings(tmp_path))
    candidate = str(tmp_path / escape) if escape.startswith("..") else escape
    with pytest.raises(CommandNotAllowedError, match="outside the folders"):
        runner._resolve_directory(candidate)


def test_a_file_is_not_a_directory(tmp_path):
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    runner = DevRunner(_settings(tmp_path))
    with pytest.raises(CommandExecutionError, match="not a directory"):
        runner._resolve_directory("notes.txt")


# -- actually running something -------------------------------------------


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


@pytest.mark.skipif(not _git_available(), reason="git is not installed")
def test_git_status_runs_in_a_real_repository(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True, timeout=30)
    result = DevRunner(_settings(tmp_path)).run("git.status")

    assert result.succeeded is True
    assert result.operation == "git.status"
    assert result.exit_code == 0


@pytest.mark.skipif(not _git_available(), reason="git is not installed")
def test_a_failing_operation_reports_rather_than_raises(tmp_path):
    # Not a git repository: git exits non-zero. That is a result to report, not
    # an exception — the user asked a question and got an answer.
    result = DevRunner(_settings(tmp_path)).run("git.status")
    assert result.succeeded is False
    assert result.exit_code != 0


# -- output truncation -----------------------------------------------------


def test_short_output_is_untouched():
    text, truncated = _tail("all good")
    assert text == "all good"
    assert truncated is False


def test_long_output_keeps_the_tail():
    # A pytest summary, a lint count and a git result all live at the end.
    body = "".join(f"line {n}\n" for n in range(50_000))
    text, truncated = _tail(body)

    assert truncated is True
    assert len(text) < len(body)
    assert text.endswith("line 49999\n")
    assert len(text) <= MAX_OUTPUT_CHARS + 100


# -- code assistant --------------------------------------------------------


def _assistant(tmp_path: Path) -> CodeAssistant:
    return CodeAssistant(FakeProvider(), _settings(tmp_path))


def test_secrets_files_are_never_sent_to_a_model(tmp_path):
    (tmp_path / ".env").write_text("QUAINEX_ANTHROPIC_API_KEY=sk-ant-real", encoding="utf-8")
    with pytest.raises(CommandNotAllowedError, match="will not be sent"):
        _assistant(tmp_path)._read_source(str(tmp_path / ".env"))


def test_binaries_are_refused(tmp_path):
    (tmp_path / "thing.exe").write_bytes(b"MZ\x00\x00")
    with pytest.raises(CommandNotAllowedError, match="not a recognised source"):
        _assistant(tmp_path)._read_source(str(tmp_path / "thing.exe"))


def test_files_outside_the_roots_are_refused(tmp_path):
    with pytest.raises(CommandNotAllowedError, match="outside the folders"):
        _assistant(tmp_path)._read_source("C:\\Windows\\System32\\drivers\\etc\\hosts")


def test_oversized_source_is_refused(tmp_path):
    big = tmp_path / "huge.py"
    big.write_text("# padding\n" * 20_000, encoding="utf-8")
    with pytest.raises(CommandExecutionError, match="limit is"):
        _assistant(tmp_path)._read_source(str(big))


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(CommandExecutionError, match="No file at"):
        _assistant(tmp_path)._read_source(str(tmp_path / "nope.py"))


def test_source_file_reads_cleanly(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("print('hello')\n", encoding="utf-8")

    text, resolved = _assistant(tmp_path)._read_source(str(source))
    assert "hello" in text
    assert resolved == source.resolve()
