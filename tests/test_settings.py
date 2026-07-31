"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quainex.config.settings import REPO_ROOT, Environment, Settings


def _make(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


def test_defaults_are_development_and_local_only():
    settings = _make()
    assert settings.environment is Environment.DEV
    assert settings.host == "127.0.0.1", "must not bind publicly before Phase 6 auth"
    assert settings.is_production is False


def test_reload_is_off_by_default():
    # The always-on autostart must be a single process. uvicorn's reloader runs a
    # supervisor plus a worker, and two such pairs is how two Telegram bridges came
    # to fight over one bot token. So reload is an explicit opt-in, even in dev.
    assert _make().reload is False
    assert _make(reload=True).reload is True


def test_production_forces_debug_off():
    # Even when an operator explicitly sets debug, production must win.
    settings = _make(environment="prod", debug=True)
    assert settings.is_production is True
    assert settings.debug is False


def test_relative_log_dir_is_anchored_to_repo_root():
    settings = _make(log_dir="logs")
    assert settings.log_dir.is_absolute()
    assert settings.log_dir == REPO_ROOT / "logs"


def test_absolute_log_dir_is_left_alone(tmp_path: Path):
    settings = _make(log_dir=tmp_path / "elsewhere")
    assert settings.log_dir == tmp_path / "elsewhere"


def test_api_key_is_not_exposed_by_repr_or_str():
    settings = _make(anthropic_api_key="sk-ant-super-secret")
    rendered = f"{settings!r} {settings}"
    assert "sk-ant-super-secret" not in rendered
    # The value is still retrievable deliberately.
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-super-secret"


@pytest.mark.parametrize(
    ("key", "expected"),
    [(None, False), ("", False), ("   ", False), ("sk-ant-abc", True)],
)
def test_has_ai_credentials(key: str | None, expected: bool):
    assert _make(anthropic_api_key=key).has_ai_credentials is expected


def test_unknown_quainex_variable_is_rejected():
    # A typo in .env should fail loudly at startup, not silently use a default.
    with pytest.raises(ValidationError):
        _make(totally_unknown_option="x")


def test_port_must_be_in_valid_range():
    with pytest.raises(ValidationError):
        _make(port=0)
    with pytest.raises(ValidationError):
        _make(port=70000)
