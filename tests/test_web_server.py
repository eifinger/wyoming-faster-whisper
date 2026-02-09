"""Tests for optional web server helpers."""

import json

from wyoming_faster_whisper.speaker_identifier import SpeakerMatch
from wyoming_faster_whisper.web_server import (
    _WEB_PAGE_HTML,
    _build_wyoming_payload,
    _collect_reference_samples,
)


def test_collect_reference_samples_from_dirs_and_files(tmp_path) -> None:
    alice_dir = tmp_path / "alice"
    alice_dir.mkdir()
    (alice_dir / "sample1.wav").write_bytes(b"test")
    (alice_dir / "note.txt").write_text("ignore")

    (tmp_path / "bob.wav").write_bytes(b"test")

    samples = _collect_reference_samples(tmp_path)
    assert sorted(samples) == ["alice", "bob"]
    assert len(samples["alice"]) == 1
    assert len(samples["bob"]) == 1


def test_build_wyoming_payload_context_mode() -> None:
    payload = _build_wyoming_payload(
        text="turn on the light",
        speaker_match=SpeakerMatch("alice", 0.82),
        output_mode="context",
        unknown_name=None,
    )

    assert payload["text"] == "turn on the light"
    assert payload["context"]["speaker"]["name"] == "alice"
    assert payload["context"]["speaker"]["score"] == 0.82


def test_build_wyoming_payload_json_text_mode() -> None:
    payload = _build_wyoming_payload(
        text="turn on the light",
        speaker_match=SpeakerMatch(None, 0.41),
        output_mode="json-text",
        unknown_name="guest",
    )

    transcript_payload = json.loads(payload["text"])
    assert transcript_payload["text"] == "turn on the light"
    assert transcript_payload["speaker"]["name"] == "guest"
    assert transcript_payload["speaker"]["score"] == 0.41
    assert payload["context"] is None


def test_web_page_includes_microphone_recording_controls() -> None:
    assert "MediaRecorder" in _WEB_PAGE_HTML
    assert "toggleTranscribeRecording" in _WEB_PAGE_HTML
    assert "toggleSpeakerRecording" in _WEB_PAGE_HTML
