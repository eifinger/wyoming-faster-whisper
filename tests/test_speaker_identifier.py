"""Tests for speaker identification."""

import pickle

import numpy as np
import pytest

from wyoming_faster_whisper import speaker_identifier
from wyoming_faster_whisper.speaker_identifier import (
    SpeakerIdentifier,
    ensure_speaker_dependencies,
    load_embeddings,
    resolve_speaker_device,
)


def test_load_embeddings_success(tmp_path) -> None:
    embeddings_path = tmp_path / "embeddings.pkl"
    raw_embeddings = {
        "bob": np.array([0.0, 2.0], dtype=np.float32),
        "alice": np.array([3.0, 0.0], dtype=np.float32),
    }

    with open(embeddings_path, "wb") as embeddings_file:
        pickle.dump(raw_embeddings, embeddings_file)

    loaded = load_embeddings(embeddings_path)
    assert list(loaded.keys()) == ["alice", "bob"]
    assert np.isclose(np.linalg.norm(loaded["alice"]), 1.0)
    assert np.isclose(np.linalg.norm(loaded["bob"]), 1.0)


def test_load_embeddings_invalid(tmp_path) -> None:
    embeddings_path = tmp_path / "embeddings.pkl"
    with open(embeddings_path, "wb") as embeddings_file:
        pickle.dump(["not", "a", "mapping"], embeddings_file)

    with pytest.raises(ValueError):
        load_embeddings(embeddings_path)


def test_identify_threshold_match(monkeypatch) -> None:
    identifier = SpeakerIdentifier(
        embeddings={
            "alice": np.array([1.0, 0.0], dtype=np.float32),
            "bob": np.array([0.0, 1.0], dtype=np.float32),
        },
        threshold=0.5,
        include_score=True,
    )

    monkeypatch.setattr(
        identifier,
        "_embed_wav",
        lambda _: np.array([0.9, 0.1], dtype=np.float32),
    )

    match = identifier.identify("test.wav")
    assert match.name == "alice"
    assert match.score is not None
    assert match.score > 0.89


def test_identify_unknown_below_threshold(monkeypatch) -> None:
    identifier = SpeakerIdentifier(
        embeddings={
            "alice": np.array([1.0, 0.0], dtype=np.float32),
            "bob": np.array([0.0, 1.0], dtype=np.float32),
        },
        threshold=0.9,
        include_score=True,
    )

    monkeypatch.setattr(
        identifier,
        "_embed_wav",
        lambda _: np.array([0.6, 0.8], dtype=np.float32),
    )

    match = identifier.identify("test.wav")
    assert match.name is None
    assert match.score is not None
    assert 0.79 < match.score < 0.81


def test_resolve_speaker_device_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        speaker_identifier,
        "_is_device_available",
        lambda device: device == "cpu",
    )

    assert resolve_speaker_device("cuda") == "cpu"
    assert resolve_speaker_device("auto") == "cpu"


def test_ensure_speaker_dependencies_pkg_resources_error(monkeypatch) -> None:
    def _import_module(name: str):
        if name == "resemblyzer":
            error = ModuleNotFoundError("No module named 'pkg_resources'")
            error.name = "pkg_resources"
            raise error
        return object()

    monkeypatch.setattr(speaker_identifier.importlib, "import_module", _import_module)

    with pytest.raises(RuntimeError, match="pkg_resources"):
        ensure_speaker_dependencies()


def test_ensure_speaker_dependencies_success(monkeypatch) -> None:
    monkeypatch.setattr(
        speaker_identifier.importlib,
        "import_module",
        lambda _: object(),
    )

    ensure_speaker_dependencies()
