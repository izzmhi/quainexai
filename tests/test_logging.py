"""Tests for log redaction.

The log file is the artefact most likely to be pasted into a bug report or a chat
window, so what it does and does not contain is a security property rather than a
formatting preference.
"""

from __future__ import annotations

from quainex.core.logging import _REDACTED, _redact_sensitive


def _scrub(**fields: object) -> dict[str, object]:
    """Run the redaction processor over one event.

    Args:
        **fields: The event payload.

    Returns:
        The scrubbed payload.
    """
    return dict(_redact_sensitive(None, "info", dict(fields)))  # type: ignore[arg-type]


def test_credentials_are_redacted_however_they_are_named():
    scrubbed = _scrub(
        api_key="sk-real-value",
        ANTHROPIC_API_KEY="sk-ant-real",
        telegram_bot_token="123:real",
        auth_secret="s3cret",
        password="hunter2",
        authorization="Bearer real",
        credential="real",
    )

    assert all(value == _REDACTED for value in scrubbed.values())


def test_a_value_is_never_partially_revealed():
    """No prefix, no length, no first four characters.

    For most vendors the prefix identifies the account tier, which is enough to
    confirm a guessed key.
    """
    scrubbed = _scrub(api_key="sk-ant-api03-abcdefghijklmnop")

    assert "sk-ant" not in str(scrubbed)
    assert "abcdefg" not in str(scrubbed)


def test_ordinary_fields_are_untouched():
    scrubbed = _scrub(intent="screenshot", target=None, confidence=0.97)

    assert scrubbed == {"intent": "screenshot", "target": None, "confidence": 0.97}


def test_token_counts_survive_because_they_are_not_secrets():
    """The false positive this allowlist exists for.

    "tokens_saved" matches the ``token`` marker, so a plain integer arrived in the
    log as "***REDACTED***" — and the entire purpose of logging it is to read the
    figure.
    """
    scrubbed = _scrub(tokens_saved=1650, max_tokens=768, prompt_tokens=42)

    assert scrubbed == {"tokens_saved": 1650, "max_tokens": 768, "prompt_tokens": 42}


def test_the_allowlist_is_exact_not_a_substring_rule():
    """Otherwise "max_tokens_key" would inherit an exemption it never earned.

    The blunt rule stays blunt on purpose; only the named exceptions escape it.
    """
    scrubbed = _scrub(tokens_saved_api_key="secret", my_max_tokens_token="secret")

    assert all(value == _REDACTED for value in scrubbed.values())
