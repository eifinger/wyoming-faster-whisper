# Speaker Identification Implementation Plan

## 1) Goals and Scope

### Goals
- Add optional speaker identification to `wyoming-faster-whisper` while keeping upstream architecture and behavior intact.
- Keep Home Assistant Voice Pipeline compatibility by default.
- Support downstream LLM callers that may prefer structured JSON output in transcript text.
- Keep speaker dependencies optional at package level.
- Include speaker dependencies in Docker image by default.

### Non-goals
- No speaker diarization (multiple speakers segmented within one utterance).
- No retraining pipeline in this repo.
- No protocol changes in Wyoming core.

---

## 2) Confirmed Design Decisions

1. **Output format**
   - Support both output modes:
     - `context` (default): plain transcript in `Transcript.text`; speaker metadata in `Transcript.context`.
     - `json-text`: JSON string in `Transcript.text` for consumers that only read text.

2. **Unknown speaker**
   - Default unknown speaker value is `null` (not `"guest"`).
   - Allow optional string override for `json-text` mode via CLI.

3. **Backend scope**
   - Speaker identification works for **all STT backends** (faster-whisper, transformers, sherpa/parakeet, onnx-asr).

4. **Dependency strategy**
   - Speaker dependencies are in optional extra: `.[speaker]`.

5. **Docker strategy**
   - Docker image includes speaker dependencies by default.

---

## 3) High-Level Architecture

## Core principle
Speaker identification is a post-capture process that runs in parallel with STT on the same final WAV file (`AudioStop` path). This is backend-agnostic.

### New components
- `wyoming_faster_whisper/speaker_identifier.py` (new)
  - Load embeddings from disk.
  - Lazy import speaker libraries.
  - Compute embedding and best match.
  - Return structured result with score + match decision.

- `script/enroll_speakers.py` (new)
  - Build embeddings file from reference audio clips.
  - Stable file format and validation.

### Integration point
- `wyoming_faster_whisper/dispatch_handler.py`
  - On `AudioStop`, run transcription and speaker ID concurrently.
  - Emit response according to `--speaker-output-mode`.

---

## 4) CLI and Config Plan

Add arguments in `wyoming_faster_whisper/__main__.py`:

- `--embeddings-file PATH`
  - Enables speaker ID when provided.
- `--speaker-threshold FLOAT` (default to be tuned; initial proposal: `0.5`)
  - Minimum similarity to accept a speaker match.
- `--speaker-device auto|cpu|cuda|mps` (default `auto`)
  - Device for speaker encoder.
- `--speaker-output-mode context|json-text` (default `context`)
  - Output contract selection.
- `--speaker-unknown-name TEXT` (optional)
  - Used only in `json-text` mode; if omitted, unknown remains `null`.
- `--speaker-include-score` (bool flag; default `true` behavior documented)
  - If enabled, include similarity score in output metadata.

Validation rules:
- If `--embeddings-file` is set but file missing/unreadable: fail startup with clear message.
- If `--speaker-*` args are set without `--embeddings-file`: warn and ignore (or fail fast; see implementation detail below).
- `speaker` mode should not alter transcription path when disabled.

Implementation preference:
- Fail fast on invalid `--embeddings-file`.
- For extra speaker flags without embeddings file: log warning and proceed without speaker ID.

---

## 5) Output Contract

### Mode A (default): `context`
`Transcript` event payload:
- `text`: plain transcript (unchanged behavior for HA)
- `context`: merge existing context (if any) +

```json
{
  "speaker": {
    "name": "alice",
    "score": 0.82
  }
}
```

Unknown speaker:

```json
{
  "speaker": {
    "name": null,
    "score": 0.41
  }
}
```

### Mode B: `json-text`
`Transcript.text` is JSON string:

```json
{"text":"turn on the light","speaker":{"name":"alice","score":0.82}}
```

Unknown speaker uses `null` unless `--speaker-unknown-name` is set.

Compatibility notes:
- Default mode is HA-safe.
- `json-text` exists for simple clients that only consume plain text field.

---

## 6) File-by-File Implementation Tasks

## A. `pyproject.toml`
- Add optional dependency group:
  - `speaker = [...]`
- Proposed deps:
  - `resemblyzer==0.1.3`
  - `librosa`
  - `soundfile`
  - `webrtcvad-wheels` (to avoid Python 3.13 `pkg_resources` issue from legacy `webrtcvad`)
- Keep base install unchanged.

## B. `script/setup`
- Add `--speaker` flag.
- Include `speaker` in extras list when selected.

## C. `Dockerfile`
- Install with extra `speaker` included by default:
  - `-e '.[zeroconf,transformers,sherpa,onnx-asr,speaker]'`
- Keep existing extras present.
- Ensure required system libs for audio are installed if needed (`ffmpeg`, `libsndfile1`, etc., if runtime requires).

## D. `wyoming_faster_whisper/speaker_identifier.py` (new)
Implement:
- `SpeakerMatch` dataclass
  - `name: Optional[str]`
  - `score: Optional[float]`
- `SpeakerIdentifier` class
  - init params: embeddings map, threshold, device, include_score
  - lazy init of `VoiceEncoder`
  - `identify(wav_path) -> SpeakerMatch`
- helper functions:
  - `load_embeddings(path) -> Dict[str, np.ndarray]`
  - optional embedding normalization sanity checks

Behavior:
- If inference fails for request: log error, return unknown (`name=None`).
- Do not fail whole transcription request due to speaker inference failure.

## E. `wyoming_faster_whisper/models.py`
- No STT logic changes required.
- Optionally add small config holder if cleaner for passing speaker settings.

## F. `wyoming_faster_whisper/__main__.py`
- Parse new CLI args.
- Initialize speaker subsystem if embeddings configured.
- Pass speaker config/instance into `DispatchEventHandler`.
- Startup logs should clearly indicate:
  - speaker ID enabled/disabled
  - embeddings file
  - threshold/device/output mode

## G. `wyoming_faster_whisper/dispatch_handler.py`
- Extend constructor with optional speaker identifier + output mode config.
- On `AudioStop`:
  - Run STT and speaker inference concurrently.
  - Keep request completion even if speaker task fails.
- Build transcript event according to output mode.
- Preserve existing language handling and transcriber lifecycle reset.

## H. `script/enroll_speakers.py` (new)
CLI:
- `--reference-dir DIR` (required)
- `--output FILE` (required)
- `--device auto|cpu|cuda|mps` (default auto)
- `--debug`

Behavior:
- Accept `.wav` at minimum (optional `.mp3` support only if dependencies are explicit).
- Speaker name derived from filename stem.
- Produce embeddings pickle with deterministic mapping.
- Print summary (speakers enrolled, skipped files, duration warnings).

## I. `README.md`
- Add "Speaker Identification" section:
  - install options (`script/setup --speaker`)
  - enrollment workflow
  - run examples for both output modes
  - Home Assistant recommendation: keep default `context`
- Document output payloads and unknown behavior.

## J. `CHANGELOG.md`
- Add unreleased entry for speaker ID feature, CLI additions, and output modes.

---

## 7) Testing Plan

## Unit tests (new: `tests/test_speaker_identifier.py`)
- Load embeddings success/failure.
- Similarity and threshold behavior.
- Unknown speaker when below threshold.
- Device selection fallback behavior (mocked).

## Integration tests (`tests/test_faster_whisper.py` additions)
- Existing baseline transcription test remains unchanged.
- New tests (mock speaker module where needed):
  1. `context` mode: transcript text unchanged + speaker metadata in context.
  2. `json-text` mode: transcript text contains valid JSON schema.
  3. Speaker failure path: transcript still returned.
  4. Feature disabled path: exact old behavior.

Test reliability strategy:
- Avoid model-heavy speaker inference in CI by mocking encoder output.
- Keep one optional/manual smoke test for real embeddings outside CI.

---

## 8) Performance and Reliability Considerations

- Parallelization at `AudioStop` reduces added latency.
- Speaker model initialized once per process; reused across requests.
- Protect speaker inference with exception handling so STT remains robust.
- Keep all new behavior opt-in via `--embeddings-file`.

---

## 9) Rollout / Migration

1. Merge feature behind optional config.
2. Release with docs and examples.
3. For HA users: recommend default `context` mode.
4. For LLM wrappers: optionally switch to `json-text`.

Backward compatibility:
- No behavior change when speaker flags are not used.

---

## 10) Implementation Sequence (PR-friendly)

### PR 1: Plumbing + package extras
- `pyproject.toml`, `script/setup`, CLI arg parsing scaffolding.
- No runtime behavior change yet.

### PR 2: Core speaker module
- Add `speaker_identifier.py` + unit tests.

### PR 3: Dispatch integration
- Parallel STT + speaker inference.
- Output mode implementation.
- Integration tests (mock-based).

### PR 4: Enrollment tooling + docs
- Add `script/enroll_speakers.py`.
- README and changelog updates.

### PR 5: Docker update
- Include `speaker` extra by default.
- Validate container runtime.

---

## 11) Acceptance Criteria

- [ ] Server runs unchanged without speaker args.
- [ ] With `--embeddings-file`, speaker metadata is produced.
- [ ] Default mode keeps `Transcript.text` plain text.
- [ ] `json-text` mode emits valid JSON with transcript + speaker.
- [ ] Unknown speaker defaults to `null`.
- [ ] Speaker inference failure does not fail transcription.
- [ ] CI tests pass with speaker logic covered.
- [ ] Docker image works with speaker dependencies installed.

---

## 12) Notes for Threshold Tuning

- Initial default: `0.5`.
- Add guidance in README to tune per environment (microphone/noise/domain).
- Consider future optional helper command to evaluate score distributions on validation clips.
