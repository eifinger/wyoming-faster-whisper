"""Speaker identification utilities."""

import importlib
import logging
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

_LOGGER = logging.getLogger(__name__)


def _speaker_dependency_error_message(err: Exception) -> str:
    if isinstance(err, ModuleNotFoundError) and (err.name == "pkg_resources"):
        return (
            "Speaker dependencies are installed but incompatible: "
            "missing 'pkg_resources' (usually setuptools>=81). "
            "Install with the 'speaker' extra and pin setuptools<81."
        )

    return (
        "Speaker dependencies are missing or broken. "
        "Install with the 'speaker' extra."
    )


def ensure_speaker_dependencies() -> None:
    """Validate speaker runtime dependencies and raise clear errors."""
    try:
        importlib.import_module("resemblyzer")
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError(_speaker_dependency_error_message(err)) from err


@dataclass
class SpeakerMatch:
    """Result of speaker identification for an utterance."""

    name: Optional[str]
    score: Optional[float]


def _normalize_embedding(
    embedding: np.ndarray, speaker_name: Optional[str] = None
) -> np.ndarray:
    """Normalize embedding to unit length."""
    if embedding.ndim != 1:
        raise ValueError("Embedding must be 1-dimensional")

    norm = np.linalg.norm(embedding)
    if norm <= 0:
        if speaker_name:
            raise ValueError(f"Embedding for '{speaker_name}' has zero norm")

        raise ValueError("Embedding has zero norm")

    return embedding / norm


def _is_device_available(device: str) -> bool:
    """Return true if a torch device is available."""
    if device == "cpu":
        return True

    try:
        import torch
    except Exception:
        return False

    if device == "cuda":
        return bool(torch.cuda.is_available())

    if device == "mps":
        return bool(
            hasattr(torch.backends, "mps")
            and (torch.backends.mps is not None)
            and torch.backends.mps.is_available()
        )

    return False


def _detect_best_device() -> str:
    """Detect the best available speaker device."""
    if _is_device_available("cuda"):
        return "cuda"

    if _is_device_available("mps"):
        return "mps"

    return "cpu"


def resolve_speaker_device(device: str) -> str:
    """Resolve requested device with graceful fallback."""
    if device == "auto":
        return _detect_best_device()

    if not _is_device_available(device):
        _LOGGER.warning(
            "Speaker device '%s' is unavailable; falling back to cpu", device
        )
        return "cpu"

    return device


def load_embeddings(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load and validate speaker embeddings from pickle file."""
    embeddings_path = Path(path)
    with open(embeddings_path, "rb") as embeddings_file:
        raw_embeddings = pickle.load(embeddings_file)

    if not isinstance(raw_embeddings, Mapping):
        raise ValueError(
            "Embeddings file must contain a mapping of speaker name to embedding"
        )

    validated_embeddings: Dict[str, np.ndarray] = {}
    for name in sorted(raw_embeddings):
        if (not isinstance(name, str)) or (not name.strip()):
            raise ValueError("Speaker names must be non-empty strings")

        embedding = np.asarray(raw_embeddings[name], dtype=np.float32)
        validated_embeddings[name] = _normalize_embedding(embedding, speaker_name=name)

    if not validated_embeddings:
        raise ValueError("Embeddings file is empty")

    return validated_embeddings


class SpeakerIdentifier:
    """Identify speaker from an utterance WAV file."""

    def __init__(
        self,
        embeddings: Mapping[str, np.ndarray],
        threshold: float = 0.5,
        device: str = "auto",
        include_score: bool = True,
    ) -> None:
        self._embeddings = {
            name: _normalize_embedding(np.asarray(embedding, dtype=np.float32), name)
            for name, embedding in embeddings.items()
        }
        if not self._embeddings:
            raise ValueError("At least one embedding is required")

        if (threshold < -1.0) or (threshold > 1.0):
            raise ValueError("Speaker threshold must be between -1.0 and 1.0")

        self._threshold = threshold
        self._device = resolve_speaker_device(device)
        self._include_score = include_score

        self._encoder: Optional[Any] = None
        self._encoder_lock = threading.Lock()

    def _get_encoder(self) -> Any:
        """Lazy-load voice encoder."""
        if self._encoder is not None:
            return self._encoder

        with self._encoder_lock:
            if self._encoder is not None:
                return self._encoder

            from resemblyzer import VoiceEncoder

            try:
                self._encoder = VoiceEncoder(device=self._device)
            except Exception:
                if self._device != "cpu":
                    _LOGGER.warning(
                        "Failed to initialize speaker encoder on %s; retrying on cpu",
                        self._device,
                    )
                    self._device = "cpu"
                    self._encoder = VoiceEncoder(device=self._device)
                else:
                    raise

        return self._encoder

    def _embed_wav(self, wav_path: Union[str, Path]) -> np.ndarray:
        """Embed utterance from wav path."""
        from resemblyzer import preprocess_wav

        encoder = self._get_encoder()
        wav = preprocess_wav(str(wav_path))
        embedding = encoder.embed_utterance(wav)
        return _normalize_embedding(np.asarray(embedding, dtype=np.float32))

    def _best_match(self, embedding: np.ndarray) -> SpeakerMatch:
        """Find closest speaker embedding."""
        best_name: Optional[str] = None
        best_score: Optional[float] = None

        for name, enrolled_embedding in self._embeddings.items():
            score = float(np.dot(embedding, enrolled_embedding))
            if (best_score is None) or (score > best_score):
                best_name = name
                best_score = score

        if best_score is None:
            return SpeakerMatch(name=None, score=None)

        if best_score < self._threshold:
            return SpeakerMatch(
                name=None,
                score=best_score if self._include_score else None,
            )

        return SpeakerMatch(
            name=best_name,
            score=best_score if self._include_score else None,
        )

    def identify(self, wav_path: Union[str, Path]) -> SpeakerMatch:
        """Identify the speaker for a WAV file."""
        try:
            embedding = self._embed_wav(wav_path)
            return self._best_match(embedding)
        except Exception:
            _LOGGER.exception("Speaker identification failed for %s", wav_path)
            return SpeakerMatch(name=None, score=None)
