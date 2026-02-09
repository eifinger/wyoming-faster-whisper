"""Event handler for clients of the server."""

import asyncio
import json
import logging
import os
import tempfile
import wave
from typing import Any, Dict, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .const import Transcriber
from .models import ModelLoader
from .speaker_identifier import SpeakerIdentifier, SpeakerMatch

_LOGGER = logging.getLogger(__name__)


class DispatchEventHandler(AsyncEventHandler):
    """Dispatches to appropriate transcriber."""

    def __init__(
        self,
        wyoming_info: Info,
        loader: ModelLoader,
        speaker_identifier: Optional[SpeakerIdentifier] = None,
        speaker_output_mode: str = "context",
        speaker_unknown_name: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()

        self._loader = loader
        self._transcriber: Optional[Transcriber] = None
        self._transcriber_future: Optional[asyncio.Future] = None
        self._language: Optional[str] = None

        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

        self._audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)

        self._speaker_identifier = speaker_identifier
        self._speaker_output_mode = speaker_output_mode
        self._speaker_unknown_name = speaker_unknown_name

    def _speaker_payload(self, speaker_match: SpeakerMatch) -> Dict[str, Any]:
        speaker_name = speaker_match.name
        if (
            (speaker_name is None)
            and (self._speaker_output_mode == "json-text")
            and (self._speaker_unknown_name is not None)
        ):
            speaker_name = self._speaker_unknown_name

        payload: Dict[str, Any] = {"name": speaker_name}
        if speaker_match.score is not None:
            payload["score"] = speaker_match.score

        return payload

    def _build_transcript(
        self, text: str, speaker_match: Optional[SpeakerMatch]
    ) -> Transcript:
        if (self._speaker_identifier is None) or (speaker_match is None):
            return Transcript(text=text)

        speaker_payload = self._speaker_payload(speaker_match)
        if self._speaker_output_mode == "json-text":
            json_payload = {"text": text, "speaker": speaker_payload}
            return Transcript(text=json.dumps(json_payload, separators=(",", ":")))

        context: Dict[str, Any] = {"speaker": speaker_payload}
        return Transcript(text=text, context=context)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            # Audio is saved to a WAV file for transcription later.
            # None of the underlying models support streaming.
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)

            if (self._transcriber is None) and (self._transcriber_future is None):
                # Load the transcriber in the background.
                # Hopefully it's ready by the time the audio stops.
                self._transcriber_future = asyncio.create_task(
                    self._loader.load_transcriber(self._language)
                )

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stoppped")

            if self._transcriber is None:
                # Get transcriber that was loading in the background
                assert self._transcriber_future is not None
                self._transcriber = await self._transcriber_future

            assert self._transcriber is not None
            assert self._wav_file is not None

            self._wav_file.close()
            self._wav_file = None

            # Do transcription and speaker identification in parallel.
            transcribe_task = asyncio.create_task(
                asyncio.to_thread(
                    self._transcriber.transcribe,
                    self._wav_path,
                    self._language,
                    beam_size=self._loader.beam_size,
                    initial_prompt=self._loader.initial_prompt,
                )
            )

            speaker_task = None
            if self._speaker_identifier is not None:
                speaker_task = asyncio.create_task(
                    asyncio.to_thread(self._speaker_identifier.identify, self._wav_path)
                )

            text = await transcribe_task

            speaker_match: Optional[SpeakerMatch] = None
            if speaker_task is not None:
                try:
                    speaker_match = await speaker_task
                except Exception as err:
                    _LOGGER.error("Speaker identification task failed: %s", err)

            _LOGGER.info(text)

            transcript = self._build_transcript(text, speaker_match)
            await self.write_event(transcript.event())
            _LOGGER.debug("Completed request")

            # Reset
            self._language = None
            self._transcriber = None

            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._language = transcribe.language or self._loader.preferred_language
            _LOGGER.debug("Language set to %s", self._language)

            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True
