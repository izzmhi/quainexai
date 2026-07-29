"""Tests for screen and document understanding.

Nothing here captures a real screen or calls a real model. The provider is faked
and the desktop controller is faked, so what is actually verified is Quainex's
own behaviour: that screenshots do not linger on disk, that paths are contained,
and that image and document payloads are built correctly.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from quainex.config.settings import Settings
from quainex.core.exceptions import CommandNotAllowedError, ProviderError
from quainex.services.ai.anthropic_provider import (
    MAX_IMAGE_BYTES,
    AnthropicProvider,
)
from quainex.vision import ScreenAnalyst
from tests.test_commands import FakeDesktopController


class FakeVisionProvider:
    """Records what it was asked to look at instead of calling a model."""

    def __init__(self) -> None:
        self.images: list[list[Path]] = []
        self.documents: list[Path] = []
        self.questions: list[str] = []
        #: Paths recorded as they existed at call time, to prove lifetime.
        self.existed_at_call: list[bool] = []

    @property
    def name(self) -> str:
        return "fake-vision"

    @property
    def is_available(self) -> bool:
        return True

    async def look(self, *, image_paths, question, system=None, max_tokens=None) -> str:
        self.images.append(list(image_paths))
        self.questions.append(question)
        self.existed_at_call.append(all(path.is_file() for path in image_paths))
        return "a window with an error dialog"

    async def read_document(self, *, document_path, question, system=None, max_tokens=None) -> str:
        self.documents.append(document_path)
        self.questions.append(question)
        return "the document says hello"

    async def complete(self, *, messages, system=None, max_tokens=None) -> str:
        return "unused"

    async def parse(self, *, messages, output_model, system=None, max_tokens=None):
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_dir=tmp_path / "logs",
        database_path=tmp_path / "t.db",
        command_search_roots=[tmp_path],
    )


def _analyst(tmp_path: Path, provider: FakeVisionProvider | None = None) -> ScreenAnalyst:
    return ScreenAnalyst(
        provider or FakeVisionProvider(), FakeDesktopController(), _settings(tmp_path)
    )


# -- screen capture lifetime ----------------------------------------------


async def test_screen_is_captured_and_the_question_passed_through(tmp_path):
    provider = FakeVisionProvider()
    answer = await _analyst(tmp_path, provider).look_at_screen("what is this error?")

    assert answer == "a window with an error dialog"
    assert provider.questions == ["what is this error?"]
    assert len(provider.images) == 1


async def test_screenshot_does_not_survive_the_call(tmp_path):
    # A picture of the user's whole desktop must not be left lying on disk.
    provider = FakeVisionProvider()
    await _analyst(tmp_path, provider).look_at_screen("anything")

    captured = provider.images[0][0]
    assert provider.existed_at_call == [True], "the image must exist while being read"
    assert not captured.exists(), "and must be gone afterwards"


# -- path containment ------------------------------------------------------


@pytest.mark.parametrize("escape", ["C:\\Windows\\explorer.exe", "..", "../../secrets.png"])
async def test_images_outside_the_roots_are_refused(tmp_path, escape):
    candidate = str(tmp_path / escape) if escape.startswith("..") else escape
    with pytest.raises(CommandNotAllowedError, match="outside the folders"):
        await _analyst(tmp_path).look_at_image(candidate, "what is this?")


async def test_documents_outside_the_roots_are_refused(tmp_path):
    with pytest.raises(CommandNotAllowedError, match="outside the folders"):
        await _analyst(tmp_path).read_document("C:\\Windows\\notes.pdf", "summarise")


async def test_document_inside_the_roots_is_read(tmp_path):
    provider = FakeVisionProvider()
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    answer = await _analyst(tmp_path, provider).read_document(str(pdf), "summarise this")

    assert answer == "the document says hello"
    assert provider.documents == [pdf.resolve()]


# -- window enumeration (local, no model) ---------------------------------


def test_window_listing_returns_titled_windows(tmp_path):
    windows = _analyst(tmp_path).list_windows()

    # Any desktop running this test has at least one titled window; the assertion
    # is on shape rather than count, so it holds in a headless CI session too.
    assert isinstance(windows, list)
    for window in windows:
        assert window.title, "untitled windows should have been filtered out"
        assert window.process_id > 0


# -- provider payload construction ----------------------------------------


def _provider(tmp_path: Path) -> AnthropicProvider:
    return AnthropicProvider(
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            log_dir=tmp_path / "logs",
            database_path=tmp_path / "t.db",
            anthropic_api_key="sk-ant-fake-for-tests",
        )
    )


def test_image_block_encodes_the_file(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n fake")

    block = _provider(tmp_path)._image_block(image)

    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(block["source"]["data"]) == image.read_bytes()


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.png", "image/png"), ("a.jpg", "image/jpeg"), ("a.webp", "image/webp")],
)
def test_media_type_follows_the_extension(tmp_path, name, expected):
    image = tmp_path / name
    image.write_bytes(b"fake")
    assert _provider(tmp_path)._image_block(image)["source"]["media_type"] == expected


def test_unsupported_image_types_are_refused(tmp_path):
    bad = tmp_path / "diagram.bmp"
    bad.write_bytes(b"BM fake")
    with pytest.raises(ProviderError, match="not a supported image type"):
        _provider(tmp_path)._image_block(bad)


def test_missing_image_is_reported(tmp_path):
    with pytest.raises(ProviderError, match="No file at"):
        _provider(tmp_path)._image_block(tmp_path / "gone.png")


def test_oversized_image_is_refused(tmp_path):
    huge = tmp_path / "huge.png"
    huge.write_bytes(b"\x00" * (MAX_IMAGE_BYTES + 1))
    with pytest.raises(ProviderError, match="the limit is"):
        _provider(tmp_path)._image_block(huge)


async def test_looking_with_no_images_is_an_error(tmp_path):
    with pytest.raises(ProviderError, match="No images"):
        await _provider(tmp_path).look(image_paths=[], question="what is this?")
