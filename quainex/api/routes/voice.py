"""Voice endpoints.

Purpose:
    Expose the voice pipeline over HTTP, including the paths that work when the
    server has no microphone of its own.

Why there is both ``/listen`` and ``/turn``:
    ``/listen`` records on the machine running Quainex. That is right for the
    desktop, and useless for the Phase 6 phone client, whose microphone is on the
    phone. ``/turn`` takes text that was recognised elsewhere and runs the same
    pipeline — so the phone can upload audio to ``/transcribe``, or transcribe
    on-device and post the text, and get identical behaviour either way.

Architecture:
    GET  /voice/status      per-component availability
    POST /voice/transcribe  audio file  -> Transcript
    POST /voice/say         text        -> spoken aloud
    POST /voice/turn        text        -> wake gate -> Brain -> commands -> speech
    POST /voice/listen      microphone  -> the whole loop

Dependencies:
    fastapi, python-multipart, quainex.core.voice

Future improvements:
    * Stream partial transcripts over the WebSocket while the user is speaking.
    * Return synthesised audio from ``/say`` so a remote client can play it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status
from pydantic import BaseModel, Field

from quainex.api.dependencies import ContainerDep
from quainex.core.exceptions import SpeechError
from quainex.core.voice.session import VoiceTurn
from quainex.core.voice.stt import Transcript

router = APIRouter(prefix="/voice", tags=["voice"])

#: Reject uploads larger than this. A minute of 16 kHz mono WAV is ~2 MB, so
#: 25 MB is generous for speech while bounding what one request can consume.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class SayRequest(BaseModel):
    """Body of a speech-output request.

    Attributes:
        text: What to say.
    """

    text: str = Field(min_length=1, examples=["Visual Studio Code is open."])


class TurnRequest(BaseModel):
    """Body of a text-driven voice turn.

    Attributes:
        text: The recognised utterance, as if it had been spoken.
        require_wake_word: Override the configured wake-word requirement. Useful
            for a push-to-talk client, where pressing the button *is* the
            addressing gesture and repeating the name is redundant.
        confirmed: Pre-approval for whatever the utterance turns out to mean.
        speak: Whether to voice the response on the host machine.
    """

    text: str = Field(min_length=1, examples=["Quainex, open VS Code"])
    require_wake_word: bool | None = None
    confirmed: bool = False
    speak: bool = True


class ListenRequest(BaseModel):
    """Body of a microphone-driven voice turn.

    Attributes:
        require_wake_word: Override the configured wake-word requirement.
        confirmed: Pre-approval for whatever is said.
    """

    require_wake_word: bool | None = None
    confirmed: bool = False


@router.get("/status", summary="Report voice subsystem availability")
async def voice_status(container: ContainerDep) -> dict[str, object]:
    """Report which voice components are usable.

    Reported per component because voice degrades in pieces: speech output needs
    no download, while recognition needs a model that may not have arrived.

    Args:
        container: Injected application container.

    Returns:
        Availability of each component.
    """
    return container.voice.status()


@router.post(
    "/transcribe",
    response_model=Transcript,
    summary="Transcribe an uploaded audio file",
    responses={503: {"description": "Speech recognition is not installed or ready."}},
)
async def transcribe(
    container: ContainerDep,
    audio: Annotated[UploadFile, File(description="Recording to transcribe.")],
) -> Transcript:
    """Transcribe uploaded audio.

    Args:
        container: Injected application container.
        audio: The uploaded recording (WAV, MP3, M4A, FLAC and similar).

    Returns:
        The transcript.

    Raises:
        SpeechError: The upload was empty or exceeded the size limit.
    """
    payload = await audio.read()
    if not payload:
        raise SpeechError("The uploaded audio file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise SpeechError(
            f"Audio is {len(payload) // 1024 // 1024} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )

    suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="quainex-upload-") as tmp:
        # Written under a name we choose: the client-supplied filename is used
        # only for its extension, never as a path.
        target = Path(tmp) / f"upload{suffix}"
        target.write_bytes(payload)
        return await container.voice.transcribe(target)


@router.post("/say", status_code=status.HTTP_200_OK, summary="Speak text aloud")
async def say(request: SayRequest, container: ContainerDep) -> dict[str, str]:
    """Say something on the host machine's speakers.

    Args:
        request: The text to speak.
        container: Injected application container.

    Returns:
        Confirmation of what was said.
    """
    await container.voice.say(request.text)
    return {"spoken": request.text}


@router.post(
    "/turn",
    response_model=VoiceTurn,
    summary="Run a voice turn from recognised text",
    responses={
        400: {"description": "The utterance was empty or too long."},
        503: {"description": "No AI provider is configured."},
    },
)
async def turn(request: TurnRequest, container: ContainerDep) -> VoiceTurn:
    """Run the full voice pipeline over text that was recognised elsewhere.

    Args:
        request: The utterance and per-turn overrides.
        container: Injected application container.

    Returns:
        The complete trace of the turn.
    """
    transcript = Transcript(text=request.text)
    return await container.voice.handle_transcript(
        transcript,
        require_wake_word=request.require_wake_word,
        confirmed=request.confirmed,
        speak=request.speak,
    )


@router.post(
    "/listen",
    response_model=VoiceTurn,
    summary="Record from the host microphone and run a voice turn",
    responses={503: {"description": "No microphone or speech recognition available."}},
)
async def listen(request: ListenRequest, container: ContainerDep) -> VoiceTurn:
    """Record an utterance on this machine and act on it.

    Blocks until the speaker stops talking, so the request takes as long as the
    user does.

    Args:
        request: Per-turn overrides.
        container: Injected application container.

    Returns:
        The complete trace of the turn.
    """
    return await container.voice.listen_and_respond(
        require_wake_word=request.require_wake_word,
        confirmed=request.confirmed,
    )
