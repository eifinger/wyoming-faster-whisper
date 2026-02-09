"""Tests for dispatch handler speaker integration."""

import json

import pytest
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.info import Info

from wyoming_faster_whisper.dispatch_handler import DispatchEventHandler
from wyoming_faster_whisper.speaker_identifier import SpeakerMatch


class _DummyWriter:
    def write(self, data) -> None:  # pragma: no cover - required by protocol
        _ = data

    async def drain(self) -> None:  # pragma: no cover - required by protocol
        return None

    def close(self) -> None:  # pragma: no cover - required by protocol
        return None


class _DummyTranscriber:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(
        self,
        wav_path,
        language,
        beam_size: int = 5,
        initial_prompt=None,
    ) -> str:
        _ = wav_path, language, beam_size, initial_prompt
        return self._text


class _DummyLoader:
    preferred_language = None
    beam_size = 5
    initial_prompt = None

    def __init__(self, text: str) -> None:
        self._transcriber = _DummyTranscriber(text)

    async def load_transcriber(self, language=None):
        _ = language
        return self._transcriber


class _DummySpeakerIdentifier:
    def __init__(self, match: SpeakerMatch) -> None:
        self._match = match

    def identify(self, wav_path) -> SpeakerMatch:
        _ = wav_path
        return self._match


class _FailingSpeakerIdentifier:
    def identify(self, wav_path) -> SpeakerMatch:
        _ = wav_path
        raise RuntimeError("speaker failure")


async def _run_request(handler: DispatchEventHandler):
    captured_events = []

    async def _capture(event) -> None:
        captured_events.append(event)

    handler.write_event = _capture  # type: ignore[method-assign]

    chunk = AudioChunk(rate=16000, width=2, channels=1, audio=b"\x00\x00" * 1600)
    assert await handler.handle_event(chunk.event()) is True
    assert await handler.handle_event(AudioStop().event()) is False

    assert len(captured_events) == 1
    return Transcript.from_event(captured_events[0])


@pytest.mark.asyncio
async def test_context_mode_with_speaker_metadata() -> None:
    handler = DispatchEventHandler(
        Info(),
        _DummyLoader("turn on the light"),
        speaker_identifier=_DummySpeakerIdentifier(SpeakerMatch("alice", 0.82)),
        speaker_output_mode="context",
        reader=None,
        writer=_DummyWriter(),
    )

    transcript = await _run_request(handler)
    assert transcript.text == "turn on the light"
    assert transcript.context is not None
    assert transcript.context["speaker"]["name"] == "alice"
    assert transcript.context["speaker"]["score"] == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_json_text_mode() -> None:
    handler = DispatchEventHandler(
        Info(),
        _DummyLoader("turn on the light"),
        speaker_identifier=_DummySpeakerIdentifier(SpeakerMatch("alice", 0.82)),
        speaker_output_mode="json-text",
        reader=None,
        writer=_DummyWriter(),
    )

    transcript = await _run_request(handler)
    payload = json.loads(transcript.text)
    assert payload["text"] == "turn on the light"
    assert payload["speaker"]["name"] == "alice"
    assert payload["speaker"]["score"] == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_json_text_unknown_name_override() -> None:
    handler = DispatchEventHandler(
        Info(),
        _DummyLoader("turn on the light"),
        speaker_identifier=_DummySpeakerIdentifier(SpeakerMatch(None, 0.41)),
        speaker_output_mode="json-text",
        speaker_unknown_name="guest",
        reader=None,
        writer=_DummyWriter(),
    )

    transcript = await _run_request(handler)
    payload = json.loads(transcript.text)
    assert payload["speaker"]["name"] == "guest"
    assert payload["speaker"]["score"] == pytest.approx(0.41)


@pytest.mark.asyncio
async def test_speaker_failure_does_not_fail_transcription() -> None:
    handler = DispatchEventHandler(
        Info(),
        _DummyLoader("turn on the light"),
        speaker_identifier=_FailingSpeakerIdentifier(),
        speaker_output_mode="context",
        reader=None,
        writer=_DummyWriter(),
    )

    transcript = await _run_request(handler)
    assert transcript.text == "turn on the light"


@pytest.mark.asyncio
async def test_speaker_disabled_keeps_original_behavior() -> None:
    handler = DispatchEventHandler(
        Info(),
        _DummyLoader("turn on the light"),
        speaker_identifier=None,
        reader=None,
        writer=_DummyWriter(),
    )

    transcript = await _run_request(handler)
    assert transcript.text == "turn on the light"
    assert transcript.context is None
