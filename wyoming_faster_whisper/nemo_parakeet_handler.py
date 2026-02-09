"""Code for transcription using NVIDIA NeMo Parakeet models."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from .const import Transcriber

_LOGGER = logging.getLogger(__name__)


class NemoParakeetTranscriber(Transcriber):
    """Wrapper for NeMo Parakeet ASR models."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        device: str = "auto",
    ) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        self._torch = torch
        self._device = self._resolve_device(device)
        self.device = self._device

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        _LOGGER.info("Loading NeMo model '%s' (device=%s)", model_id, self._device)
        self.model: Any = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_id,
        )

        if self._device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"

        if device == "cuda" and (not self._torch.cuda.is_available()):
            _LOGGER.warning("CUDA requested but unavailable; falling back to cpu")
            return "cpu"

        return device

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file.

        WAV file should be 16Khz mono audio.
        """
        _ = beam_size
        _ = initial_prompt
        _ = language

        results = self.model.transcribe([str(wav_path)], batch_size=1)
        if not results:
            return ""

        result = results[0]
        if hasattr(result, "text"):
            text = str(result.text)
        elif isinstance(result, dict) and ("text" in result):
            text = str(result["text"])
        else:
            text = str(result)

        return text.strip()
