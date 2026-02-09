#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "numpy",
#   "resemblyzer==0.1.3",
# ]
# ///

"""Enroll speaker embeddings from reference audio files."""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from wyoming_faster_whisper.speaker_identifier import resolve_speaker_device

_LOGGER = logging.getLogger(__name__)
_SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm <= 0:
        raise ValueError("Embedding has zero norm")

    return embedding / norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True, help="Directory with samples")
    parser.add_argument("--output", required=True, help="Output embeddings pickle file")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Encoder device (default: auto)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    reference_dir = Path(args.reference_dir)
    if not reference_dir.is_dir():
        raise SystemExit(f"Reference directory does not exist: {reference_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_speaker_device(args.device)
    _LOGGER.info("Using speaker encoder device: %s", device)
    encoder = VoiceEncoder(device=device)

    enrolled_embeddings: Dict[str, np.ndarray] = {}
    skipped = 0

    for audio_path in sorted(reference_dir.iterdir()):
        if not audio_path.is_file():
            continue

        if audio_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            _LOGGER.debug("Skipping unsupported file type: %s", audio_path)
            skipped += 1
            continue

        speaker_name = audio_path.stem.strip()
        if not speaker_name:
            _LOGGER.warning("Skipping file with empty speaker name: %s", audio_path)
            skipped += 1
            continue

        if speaker_name in enrolled_embeddings:
            _LOGGER.warning(
                "Skipping duplicate speaker name '%s' from %s", speaker_name, audio_path
            )
            skipped += 1
            continue

        try:
            wav = preprocess_wav(str(audio_path))
            duration_seconds = len(wav) / 16000.0
            if duration_seconds < 1.0:
                _LOGGER.warning(
                    "Reference clip is short for '%s' (%.2fs): %s",
                    speaker_name,
                    duration_seconds,
                    audio_path,
                )

            embedding = encoder.embed_utterance(wav)
            enrolled_embeddings[speaker_name] = _normalize_embedding(
                np.asarray(embedding, dtype=np.float32)
            )
            _LOGGER.info("Enrolled %s from %s", speaker_name, audio_path.name)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.warning("Skipping %s: %s", audio_path, err)
            skipped += 1

    if not enrolled_embeddings:
        raise SystemExit("No speakers enrolled")

    ordered_embeddings = {
        name: enrolled_embeddings[name] for name in sorted(enrolled_embeddings)
    }

    with open(output_path, "wb") as output_file:
        pickle.dump(ordered_embeddings, output_file, protocol=4)

    _LOGGER.info(
        "Wrote embeddings to %s (speakers=%s, skipped=%s)",
        output_path,
        len(ordered_embeddings),
        skipped,
    )


if __name__ == "__main__":
    main()
