"""Tests for model selection helpers."""

from wyoming_faster_whisper.const import SttLibrary
from wyoming_faster_whisper.models import guess_model


def test_guess_model_nemo() -> None:
    assert guess_model(SttLibrary.NEMO, language="en", is_arm=False) == (
        "nvidia/parakeet-tdt-0.6b-v3"
    )
