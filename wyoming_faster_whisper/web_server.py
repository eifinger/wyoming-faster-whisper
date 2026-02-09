"""Optional web UI for quick transcription and speaker testing."""

import asyncio
import importlib
import json
import logging
import pickle
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .models import ModelLoader
from .speaker_identifier import SpeakerIdentifier, SpeakerMatch, load_embeddings

_LOGGER = logging.getLogger(__name__)
_SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}


@dataclass
class WebServerConfig:
    """Configuration for the optional web server."""

    speaker_samples_dir: Path
    embeddings_file: Path
    speaker_threshold: float
    speaker_device: str
    speaker_include_score: bool
    speaker_unknown_name: Optional[str]
    speaker_output_mode: str


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return safe.strip("_")


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return safe.strip("._")


def _collect_reference_samples(reference_dir: Path) -> Dict[str, List[Path]]:
    samples: Dict[str, List[Path]] = {}

    if not reference_dir.exists():
        return samples

    for child in sorted(reference_dir.iterdir()):
        if child.is_dir():
            speaker_name = _safe_name(child.name)
            if not speaker_name:
                continue

            for audio_path in sorted(child.iterdir()):
                if (not audio_path.is_file()) or (
                    audio_path.suffix.lower() not in _SUPPORTED_AUDIO_EXTENSIONS
                ):
                    continue

                samples.setdefault(speaker_name, []).append(audio_path)
        elif child.is_file() and (child.suffix.lower() in _SUPPORTED_AUDIO_EXTENSIONS):
            speaker_name = _safe_name(child.stem)
            if not speaker_name:
                continue

            samples.setdefault(speaker_name, []).append(child)

    return samples


def _reference_summary(reference_dir: Path) -> List[Dict[str, Any]]:
    samples = _collect_reference_samples(reference_dir)
    return [
        {"name": speaker_name, "samples": len(samples[speaker_name])}
        for speaker_name in sorted(samples)
    ]


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm <= 0:
        raise ValueError("Embedding has zero norm")

    return embedding / norm


def _enroll_embeddings(
    reference_dir: Path, embeddings_file: Path, device: str
) -> Dict[str, Any]:
    samples = _collect_reference_samples(reference_dir)
    if not samples:
        raise ValueError(f"No speaker samples found in {reference_dir}")

    try:
        from .speaker_identifier import ensure_speaker_dependencies

        ensure_speaker_dependencies()
        resemblyzer = importlib.import_module("resemblyzer")
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError(
            f"Speaker enrollment dependency check failed: {err}"
        ) from err

    preprocess_wav = getattr(resemblyzer, "preprocess_wav")
    voice_encoder_class = getattr(resemblyzer, "VoiceEncoder")

    from .speaker_identifier import resolve_speaker_device

    resolved_device = resolve_speaker_device(device)
    encoder = voice_encoder_class(device=resolved_device)

    enrolled_embeddings: Dict[str, np.ndarray] = {}
    skipped_files = 0

    for speaker_name in sorted(samples):
        vectors: List[np.ndarray] = []
        for sample_path in samples[speaker_name]:
            try:
                wav = preprocess_wav(str(sample_path))
                vector = np.asarray(encoder.embed_utterance(wav), dtype=np.float32)
                vectors.append(_normalize_embedding(vector))
            except Exception as err:  # pylint: disable=broad-except
                skipped_files += 1
                _LOGGER.warning("Skipping sample %s: %s", sample_path, err)

        if vectors:
            mean_vector = np.mean(vectors, axis=0).astype(np.float32)
            enrolled_embeddings[speaker_name] = _normalize_embedding(mean_vector)

    if not enrolled_embeddings:
        raise ValueError("No valid speaker samples were enrolled")

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    ordered_embeddings = {
        name: enrolled_embeddings[name] for name in sorted(enrolled_embeddings)
    }

    with open(embeddings_file, "wb") as output_file:
        pickle.dump(ordered_embeddings, output_file, protocol=4)

    return {
        "speakers": sorted(ordered_embeddings),
        "speaker_count": len(ordered_embeddings),
        "skipped_files": skipped_files,
        "embeddings_file": str(embeddings_file),
    }


def _load_runtime_speaker_identifier(
    config: WebServerConfig,
) -> Optional[SpeakerIdentifier]:
    if not config.embeddings_file.is_file():
        return None

    embeddings = load_embeddings(config.embeddings_file)
    return SpeakerIdentifier(
        embeddings=embeddings,
        threshold=config.speaker_threshold,
        device=config.speaker_device,
        include_score=config.speaker_include_score,
    )


def _speaker_payload(
    speaker_match: Optional[SpeakerMatch],
    output_mode: str,
    unknown_name: Optional[str],
) -> Dict[str, Any]:
    speaker_name: Optional[str] = None
    score: Optional[float] = None

    if speaker_match is not None:
        speaker_name = speaker_match.name
        score = speaker_match.score

    if (speaker_name is None) and (output_mode == "json-text") and unknown_name:
        speaker_name = unknown_name

    payload: Dict[str, Any] = {"name": speaker_name}
    if score is not None:
        payload["score"] = score

    return payload


def _build_wyoming_payload(
    text: str,
    speaker_match: Optional[SpeakerMatch],
    output_mode: str,
    unknown_name: Optional[str],
) -> Dict[str, Any]:
    speaker_payload = _speaker_payload(speaker_match, output_mode, unknown_name)

    if output_mode == "json-text":
        transcript_text = json.dumps(
            {"text": text, "speaker": speaker_payload}, separators=(",", ":")
        )
        return {"text": transcript_text, "context": None}

    return {
        "text": text,
        "context": {"speaker": speaker_payload},
    }


def _convert_to_wav_16khz(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg conversion failed")


def create_web_app(loader: ModelLoader, config: WebServerConfig) -> Any:
    """Create FastAPI app lazily so web deps stay optional."""
    try:
        fastapi_module = importlib.import_module("fastapi")
        responses_module = importlib.import_module("fastapi.responses")
    except ImportError as err:
        raise RuntimeError(
            "Web server requires optional dependencies. Install with the 'web' extra."
        ) from err

    FastAPI = getattr(fastapi_module, "FastAPI")
    File = getattr(fastapi_module, "File")
    Form = getattr(fastapi_module, "Form")
    HTTPException = getattr(fastapi_module, "HTTPException")
    HTMLResponse = getattr(responses_module, "HTMLResponse")

    app = FastAPI(title="Wyoming Faster Whisper Test UI")
    app.state.loader = loader
    app.state.config = config
    app.state.speaker_identifier = None
    app.state.embeddings_mtime = None

    config.speaker_samples_dir.mkdir(parents=True, exist_ok=True)

    def refresh_identifier() -> None:
        embeddings_mtime = (
            config.embeddings_file.stat().st_mtime
            if config.embeddings_file.is_file()
            else None
        )
        if embeddings_mtime == app.state.embeddings_mtime:
            return

        app.state.embeddings_mtime = embeddings_mtime
        if embeddings_mtime is None:
            app.state.speaker_identifier = None
            return

        try:
            app.state.speaker_identifier = _load_runtime_speaker_identifier(config)
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error(
                "Failed to load embeddings from %s: %s", config.embeddings_file, err
            )
            app.state.speaker_identifier = None

    refresh_identifier()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> Any:
        return HTMLResponse(_WEB_PAGE_HTML)

    @app.get("/api/speakers")
    async def api_speakers() -> Dict[str, Any]:
        refresh_identifier()

        enrolled_speakers: List[str] = []
        if config.embeddings_file.is_file():
            try:
                enrolled_speakers = sorted(
                    load_embeddings(config.embeddings_file).keys()
                )
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error(
                    "Failed to inspect embeddings file %s: %s",
                    config.embeddings_file,
                    err,
                )

        return {
            "reference_dir": str(config.speaker_samples_dir),
            "reference_speakers": _reference_summary(config.speaker_samples_dir),
            "embeddings_file": str(config.embeddings_file),
            "enrolled_speakers": enrolled_speakers,
            "speaker_identification_enabled": app.state.speaker_identifier is not None,
        }

    @app.post("/api/speakers/upload")
    async def api_upload_speaker_sample(
        speaker_name: str = Form(...),
        audio_file: Any = File(...),
    ) -> Dict[str, Any]:
        normalized_name = _safe_name(speaker_name)
        if not normalized_name:
            raise HTTPException(status_code=400, detail="Speaker name cannot be empty")

        original_filename = audio_file.filename or "sample.wav"
        suffix = Path(original_filename).suffix.lower() or ".wav"
        if suffix not in _SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'",
            )

        speaker_dir = config.speaker_samples_dir / normalized_name
        speaker_dir.mkdir(parents=True, exist_ok=True)

        timestamp_ms = int(time.time() * 1000)
        base_name = _safe_filename(Path(original_filename).stem) or "sample"
        output_path = speaker_dir / f"{timestamp_ms}_{base_name}{suffix}"

        content = await audio_file.read()
        output_path.write_bytes(content)

        return {
            "speaker": normalized_name,
            "saved_to": str(output_path),
            "bytes": len(content),
        }

    @app.post("/api/speakers/enroll")
    async def api_enroll_speakers() -> Dict[str, Any]:
        try:
            result = await asyncio.to_thread(
                _enroll_embeddings,
                config.speaker_samples_dir,
                config.embeddings_file,
                config.speaker_device,
            )
        except Exception as err:  # pylint: disable=broad-except
            raise HTTPException(status_code=400, detail=str(err)) from err

        refresh_identifier()
        return result

    @app.post("/api/transcribe")
    async def api_transcribe(
        audio_file: Any = File(...),
        language: str = Form(""),
        output_mode: str = Form(config.speaker_output_mode),
    ) -> Dict[str, Any]:
        if output_mode not in ("context", "json-text"):
            raise HTTPException(
                status_code=400, detail="output_mode must be context or json-text"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            source_suffix = (
                Path(audio_file.filename or "audio.wav").suffix.lower() or ".wav"
            )
            source_path = temp_dir_path / f"source{source_suffix}"
            source_path.write_bytes(await audio_file.read())

            wav_path = temp_dir_path / "speech.wav"
            try:
                await asyncio.to_thread(_convert_to_wav_16khz, source_path, wav_path)
            except Exception as err:  # pylint: disable=broad-except
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to convert audio file: {err}",
                ) from err

            request_language: Optional[str] = language.strip() or None
            text = await loader.transcribe(wav_path, request_language)

            refresh_identifier()
            speaker_match: Optional[SpeakerMatch] = None
            if app.state.speaker_identifier is not None:
                speaker_match = await asyncio.to_thread(
                    app.state.speaker_identifier.identify,
                    wav_path,
                )

        wyoming_payload = _build_wyoming_payload(
            text=text,
            speaker_match=speaker_match,
            output_mode=output_mode,
            unknown_name=config.speaker_unknown_name,
        )

        return {
            "text": text,
            "speaker": _speaker_payload(
                speaker_match=speaker_match,
                output_mode=output_mode,
                unknown_name=config.speaker_unknown_name,
            ),
            "wyoming": wyoming_payload,
        }

    return app


async def run_web_server(app: Any, host: str, port: int) -> None:
    """Run FastAPI app with uvicorn."""
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ImportError as err:
        raise RuntimeError(
            "Web server requires optional dependencies. Install with the 'web' extra."
        ) from err

    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


_WEB_PAGE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Wyoming Faster Whisper - Test UI</title>
  <style>
    :root {
      --bg: #0b1020;
      --bg-2: #101833;
      --panel: #121c38cc;
      --panel-border: #3e4f88;
      --text: #eaf0ff;
      --muted: #a4b2da;
      --accent: #6aa6ff;
      --accent-2: #74f1d7;
      --danger: #ff7b9c;
      --code-bg: #0a1022;
    }

    * { box-sizing: border-box; }

    body {
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(1200px 500px at 10% -10%, #2b3d7a88, transparent),
        radial-gradient(800px 400px at 100% 0, #155a8e88, transparent),
        linear-gradient(180deg, var(--bg), var(--bg-2));
      min-height: 100vh;
      padding: 2rem 1rem;
    }

    .container {
      max-width: 1000px;
      margin: 0 auto;
    }

    .hero {
      padding: 1.4rem 1.5rem;
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      background: linear-gradient(135deg, #182750dd, #122349dd);
      box-shadow: 0 18px 40px #00000055;
      margin-bottom: 1rem;
    }

    h1, h2 {
      margin: 0 0 0.6rem;
      letter-spacing: 0.2px;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }

    section {
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      background: var(--panel);
      backdrop-filter: blur(8px);
      box-shadow: 0 10px 28px #00000045;
      padding: 1rem;
    }

    .wide {
      grid-column: 1 / -1;
    }

    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      margin-bottom: 0.6rem;
    }

    .badge {
      border: 1px solid #7daeff66;
      color: #cfe0ff;
      background: #2b4f9a55;
      border-radius: 999px;
      font-size: 0.78rem;
      padding: 0.2rem 0.6rem;
      white-space: nowrap;
    }

    .row {
      display: flex;
      gap: 0.7rem;
      flex-wrap: wrap;
      align-items: center;
    }

    input, select, button {
      border-radius: 10px;
      border: 1px solid #49639f;
      padding: 0.55rem 0.65rem;
      font-size: 0.95rem;
    }

    input, select {
      color: var(--text);
      background: #0f1833;
      min-height: 38px;
    }

    input::placeholder {
      color: #8ea2d0;
    }

    button {
      color: #071127;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      border: none;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      box-shadow: 0 6px 16px #5aa3ff55;
    }

    button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 22px #5aa3ff66;
    }

    button:active {
      transform: translateY(0);
    }

    pre {
      background: var(--code-bg);
      color: #dce7ff;
      border: 1px solid #2f3e6f;
      padding: 0.95rem;
      border-radius: 10px;
      overflow: auto;
      min-height: 140px;
      margin: 0.5rem 0 0;
    }

    .muted {
      color: var(--muted);
      margin: 0.5rem 0 0;
      min-height: 1.2rem;
    }

    audio {
      width: 100%;
      margin-top: 0.6rem;
      border-radius: 999px;
      background: #0f1833;
    }

    .note {
      margin-top: 0.5rem;
      font-size: 0.88rem;
    }

    code {
      background: #24356299;
      border: 1px solid #49639f;
      border-radius: 6px;
      padding: 0.1rem 0.35rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="hero">
      <h1>Wyoming Faster Whisper · Test UI</h1>
      <p class="subtitle">
        Train speaker identities and test speech-to-text directly in your browser.
        You can upload files or record with your microphone.
      </p>
    </header>

    <div class="grid">
      <section>
        <div class="section-title">
          <h2>1) Add speaker sample</h2>
          <span class="badge">Speaker enrollment</span>
        </div>
        <div class="row">
          <input id="speakerName" placeholder="speaker name (e.g. alice)" />
          <input id="speakerFile" type="file" accept="audio/*" />
          <button onclick="uploadSample()">Upload sample</button>
        </div>
        <div class="row" style="margin-top: 0.6rem;">
          <button id="recordSpeakerBtn" onclick="toggleSpeakerRecording()">🎙️ Record speaker sample</button>
        </div>
        <audio id="speakerPreview" controls hidden></audio>
        <p id="uploadStatus" class="muted"></p>
        <p class="muted note">Tip: record 2-3 short clips per speaker for better matching quality.</p>
      </section>

      <section>
        <div class="section-title">
          <h2>2) Enroll embeddings</h2>
          <span class="badge">Build voice vectors</span>
        </div>
        <p class="subtitle">After adding samples, generate/update <code>speakers.pkl</code>.</p>
        <div class="row" style="margin-top: 0.7rem;">
          <button onclick="enroll()">Enroll from uploaded samples</button>
        </div>
        <p id="enrollStatus" class="muted"></p>
      </section>

      <section class="wide">
        <div class="section-title">
          <h2>3) Test transcription</h2>
          <span class="badge">STT + speaker ID</span>
        </div>
        <div class="row">
          <input id="testFile" type="file" accept="audio/*" />
          <input id="language" placeholder="language (optional, e.g. en)" />
          <select id="outputMode">
            <option value="context">context</option>
            <option value="json-text">json-text</option>
          </select>
          <button onclick="transcribe()">Transcribe</button>
        </div>
        <div class="row" style="margin-top: 0.6rem;">
          <button id="recordTranscribeBtn" onclick="toggleTranscribeRecording()">🎙️ Start recording for transcription</button>
        </div>
        <audio id="transcribePreview" controls hidden></audio>
        <p id="transcribeStatus" class="muted"></p>
        <pre id="result">{}</pre>
      </section>

      <section class="wide">
        <div class="section-title">
          <h2>Current speakers</h2>
          <span class="badge">Reference + enrolled</span>
        </div>
        <div class="row">
          <button onclick="refreshSpeakers()">Refresh</button>
        </div>
        <pre id="speakers">[]</pre>
      </section>
    </div>
  </div>

  <script>
    let currentRecorder = null;
    let currentStream = null;
    let activeRecordingKind = null;
    let recordedSpeakerBlob = null;
    let recordedSpeakerFilename = null;
    let recordedTranscribeBlob = null;
    let recordedTranscribeFilename = null;

    function setStatus(elementId, message) {
      document.getElementById(elementId).textContent = message;
    }

    function resetRecordButtons() {
      document.getElementById('recordSpeakerBtn').textContent = '🎙️ Record speaker sample';
      document.getElementById('recordTranscribeBtn').textContent = '🎙️ Start recording for transcription';
    }

    function recordingSupported() {
      return !!(window.MediaRecorder && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    function pickMimeType() {
      if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) {
        return '';
      }

      const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
        'audio/mp4',
      ];

      for (const candidate of candidates) {
        if (MediaRecorder.isTypeSupported(candidate)) {
          return candidate;
        }
      }

      return '';
    }

    function extensionFromMimeType(mimeType) {
      if (mimeType.includes('webm')) return 'webm';
      if (mimeType.includes('ogg')) return 'ogg';
      if (mimeType.includes('mp4')) return 'm4a';
      return 'wav';
    }

    function stopCurrentRecorder() {
      if (currentRecorder && currentRecorder.state !== 'inactive') {
        currentRecorder.stop();
      }
    }

    function startRecording(recordingKind, onStopped, statusElementId) {
      if (!recordingSupported()) {
        setStatus(
          statusElementId,
          'Microphone recording is not supported in this browser. Use file upload instead.'
        );
        resetRecordButtons();
        return;
      }

      if (currentRecorder && currentRecorder.state !== 'inactive') {
        setStatus(statusElementId, 'Another recording is in progress. Stop it first.');
        resetRecordButtons();
        return;
      }

      navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        const chunks = [];
        const mimeType = pickMimeType();
        const options = mimeType ? { mimeType } : undefined;

        activeRecordingKind = recordingKind;
        currentStream = stream;
        currentRecorder = new MediaRecorder(stream, options);

        currentRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) {
            chunks.push(event.data);
          }
        };

        currentRecorder.onstop = () => {
          const finalMimeType = mimeType || 'audio/webm';
          const extension = extensionFromMimeType(finalMimeType);
          const blob = new Blob(chunks, { type: finalMimeType });
          const filename = `recording.${extension}`;

          if (currentStream) {
            currentStream.getTracks().forEach((track) => track.stop());
          }

          activeRecordingKind = null;
          currentRecorder = null;
          currentStream = null;
          resetRecordButtons();

          onStopped(blob, filename);
        };

        currentRecorder.onerror = (event) => {
          activeRecordingKind = null;
          setStatus(statusElementId, `Recording failed: ${event.error?.message || 'unknown error'}`);
          resetRecordButtons();
        };

        currentRecorder.start();
      }).catch((err) => {
        activeRecordingKind = null;
        setStatus(statusElementId, `Microphone access failed: ${err}`);
        resetRecordButtons();
      });
    }

    function setAudioPreview(elementId, blob) {
      const audio = document.getElementById(elementId);
      audio.src = URL.createObjectURL(blob);
      audio.hidden = false;
    }

    async function refreshSpeakers() {
      const resp = await fetch('/api/speakers');
      const data = await resp.json();
      document.getElementById('speakers').textContent = JSON.stringify(data, null, 2);
    }

    async function uploadSample() {
      const speakerName = document.getElementById('speakerName').value.trim();
      const selectedFile = document.getElementById('speakerFile').files[0];
      const file = selectedFile || recordedSpeakerBlob;
      const filename = selectedFile ? selectedFile.name : (recordedSpeakerFilename || 'sample.webm');

      if (!speakerName || !file) {
        setStatus('uploadStatus', 'Please set speaker name and choose a file or record a sample.');
        return;
      }

      const form = new FormData();
      form.append('speaker_name', speakerName);
      form.append('audio_file', file, filename);

      const resp = await fetch('/api/speakers/upload', { method: 'POST', body: form });
      const data = await resp.json();
      if (resp.ok) {
        setStatus('uploadStatus', 'Uploaded.');
        if (!selectedFile) {
          recordedSpeakerBlob = null;
          recordedSpeakerFilename = null;
        }
      } else {
        setStatus('uploadStatus', `Error: ${data.detail || JSON.stringify(data)}`);
      }
      await refreshSpeakers();
    }

    async function enroll() {
      const resp = await fetch('/api/speakers/enroll', { method: 'POST' });
      const data = await resp.json();
      document.getElementById('enrollStatus').textContent = resp.ok
        ? `Enrolled ${data.speaker_count} speakers (${data.speakers.join(', ')})`
        : `Error: ${data.detail || JSON.stringify(data)}`;
      await refreshSpeakers();
    }

    async function transcribeFileOrBlob(file, filename) {
      const form = new FormData();
      form.append('audio_file', file, filename || 'speech.webm');
      form.append('language', document.getElementById('language').value || '');
      form.append('output_mode', document.getElementById('outputMode').value);

      const resp = await fetch('/api/transcribe', { method: 'POST', body: form });
      const data = await resp.json();
      document.getElementById('result').textContent = JSON.stringify(data, null, 2);
      setStatus('transcribeStatus', resp.ok ? 'Done.' : `Error: ${data.detail || JSON.stringify(data)}`);
    }

    async function transcribe() {
      const selectedFile = document.getElementById('testFile').files[0];
      const file = selectedFile || recordedTranscribeBlob;
      const filename = selectedFile ? selectedFile.name : (recordedTranscribeFilename || 'speech.webm');
      if (!file) {
        setStatus('transcribeStatus', 'Choose a test file or record speech first.');
        return;
      }

      setStatus('transcribeStatus', 'Transcribing...');
      await transcribeFileOrBlob(file, filename);
    }

    function toggleSpeakerRecording() {
      const button = document.getElementById('recordSpeakerBtn');

      if (activeRecordingKind === 'speaker') {
        button.textContent = '🎙️ Record speaker sample';
        stopCurrentRecorder();
        return;
      }

      if (currentRecorder && currentRecorder.state !== 'inactive') {
        setStatus('uploadStatus', 'Another recording is in progress. Stop it first.');
        return;
      }

      button.textContent = '⏹ Stop recording';
      setStatus('uploadStatus', 'Recording speaker sample...');
      startRecording('speaker', (blob, filename) => {
        recordedSpeakerBlob = blob;
        recordedSpeakerFilename = filename;
        setAudioPreview('speakerPreview', blob);
        setStatus('uploadStatus', 'Recording ready. Click "Upload selected file/recording".');
        button.textContent = '🎙️ Record speaker sample';
      }, 'uploadStatus');
    }

    function toggleTranscribeRecording() {
      const button = document.getElementById('recordTranscribeBtn');

      if (activeRecordingKind === 'transcribe') {
        setStatus('transcribeStatus', 'Finishing recording...');
        stopCurrentRecorder();
        return;
      }

      if (currentRecorder && currentRecorder.state !== 'inactive') {
        setStatus('transcribeStatus', 'Another recording is in progress. Stop it first.');
        return;
      }

      button.textContent = '⏹ Stop and transcribe';
      setStatus('transcribeStatus', 'Recording from microphone...');
      startRecording('transcribe', async (blob, filename) => {
        recordedTranscribeBlob = blob;
        recordedTranscribeFilename = filename;
        setAudioPreview('transcribePreview', blob);
        setStatus('transcribeStatus', 'Transcribing microphone recording...');
        button.textContent = '🎙️ Start recording for transcription';
        await transcribeFileOrBlob(blob, filename);
      }, 'transcribeStatus');
    }

    refreshSpeakers();
  </script>
</body>
</html>
"""
