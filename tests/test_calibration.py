"""Tests for microphone calibration.

The measurement half needs a microphone and a person, so it is not tested here.
What is tested is the part that edits a file the user depends on — because writing
``.env`` wrongly is how a working configuration becomes a broken one.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def calibrate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The calibration module, with ``REPO_ROOT`` pointed at a temp directory."""
    module = importlib.import_module("scripts.calibrate_microphone")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    return module


def test_an_existing_threshold_is_replaced_not_duplicated(calibrate, tmp_path: Path):
    """Two assignments of one variable is a file that behaves unlike how it reads.

    The last one wins silently, so someone reading the first would draw the wrong
    conclusion about what their system is doing.
    """
    env = tmp_path / ".env"
    env.write_text(
        "QUAINEX_WAKE_WORD=quainex\n"
        "QUAINEX_VOICE_SILENCE_THRESHOLD=350\n"
        "QUAINEX_TTS_ENABLED=true\n",
        encoding="utf-8",
    )

    assert calibrate._write_threshold(42) is True

    text = env.read_text(encoding="utf-8")
    assert text.count("QUAINEX_VOICE_SILENCE_THRESHOLD") == 1
    assert "QUAINEX_VOICE_SILENCE_THRESHOLD=42" in text


def test_surrounding_settings_are_left_alone(calibrate, tmp_path: Path):
    """It edits one line of a file holding the user's whole configuration."""
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\nQUAINEX_WAKE_WORD=quainex\nQUAINEX_VOICE_SILENCE_THRESHOLD=350\n",
        encoding="utf-8",
    )

    calibrate._write_threshold(75)

    text = env.read_text(encoding="utf-8")
    assert "# a comment" in text
    assert "QUAINEX_WAKE_WORD=quainex" in text


def test_the_setting_is_appended_when_absent(calibrate, tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("QUAINEX_WAKE_WORD=quainex\n", encoding="utf-8")

    assert calibrate._write_threshold(60) is True

    assert "QUAINEX_VOICE_SILENCE_THRESHOLD=60" in env.read_text(encoding="utf-8")


def test_a_missing_env_file_is_reported_rather_than_created(calibrate, tmp_path: Path):
    """Creating one would produce a file with a single setting and no key.

    Quainex would then start with defaults for everything else, which looks like a
    configuration and is not one.
    """
    assert calibrate._write_threshold(60) is False
    assert not (tmp_path / ".env").exists()


def test_a_commented_out_setting_is_not_treated_as_the_live_one(calibrate, tmp_path: Path):
    """Otherwise the value gets written into a comment and has no effect."""
    env = tmp_path / ".env"
    env.write_text(
        "# QUAINEX_VOICE_SILENCE_THRESHOLD=999\nQUAINEX_WAKE_WORD=quainex\n",
        encoding="utf-8",
    )

    calibrate._write_threshold(50)

    text = env.read_text(encoding="utf-8")
    assert "# QUAINEX_VOICE_SILENCE_THRESHOLD=999" in text
    assert "\nQUAINEX_VOICE_SILENCE_THRESHOLD=50" in text
