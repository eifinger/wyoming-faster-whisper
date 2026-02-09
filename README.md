# Wyoming Faster Whisper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [faster-whisper](https://github.com/guillaumekln/faster-whisper/) speech to text system.

## Home Assistant Add-on

[![Show add-on](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=core_whisper)

[Source](https://github.com/home-assistant/addons/tree/master/whisper)

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/rhasspy/wyoming-faster-whisper.git
cd wyoming-faster-whisper
script/setup
```

To include speaker identification and web UI dependencies:

``` sh
script/setup --speaker --web
```

To run NVIDIA NeMo Parakeet models (GPU-oriented), also install NeMo deps:

``` sh
script/setup --speaker --web --nemo
```

Run a server anyone can connect to:

```sh
script/run --model tiny-int8 --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data
```

The `--model` can also be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

**NOTE**: Models are downloaded to the first `--data-dir` directory.

## Speaker Identification

Speaker identification is optional and works across all STT backends.

### 1) Enroll speaker embeddings

Create one (or more) reference clips, then build an embeddings file:

```sh
uv run script/enroll_speakers.py --reference-dir /path/to/reference_audio --output /data/speakers.pkl
```

Speaker names are taken from file stems (`alice.wav` -> `alice`).

### Embeddings file format

The embeddings file is a Python pickle containing a mapping:

- `Dict[str, numpy.ndarray]`
- key: speaker name
- value: normalized 1D embedding vector (`float32`)

Example logical structure:

```python
{
  "alice": array([...], dtype=float32),
  "bob": array([...], dtype=float32),
}
```

Quickly inspect an embeddings file:

```sh
uv run --with numpy python - <<'PY'
import pickle
from pathlib import Path

path = Path('/data/speakers.pkl')
with open(path, 'rb') as f:
    data = pickle.load(f)

print('speakers:', sorted(data))
for name in sorted(data):
    emb = data[name]
    print(name, getattr(emb, 'shape', None), getattr(emb, 'dtype', None))
PY
```

### 2) Start server with speaker identification

Default (`context`) mode keeps transcript text unchanged and adds speaker metadata in transcript context:

```sh
script/run \
  --model tiny-int8 \
  --language en \
  --uri 'tcp://0.0.0.0:10300' \
  --data-dir /data \
  --download-dir /data \
  --embeddings-file /data/speakers.pkl
```

`json-text` mode places transcript + speaker object in `Transcript.text`:

```sh
script/run \
  --model tiny-int8 \
  --language en \
  --uri 'tcp://0.0.0.0:10300' \
  --data-dir /data \
  --download-dir /data \
  --embeddings-file /data/speakers.pkl \
  --speaker-output-mode json-text
```

Available speaker options:

- `--embeddings-file` (default: `<first data dir>/speakers.pkl`)
- `--speaker-threshold` (default: `0.5`)
- `--speaker-device auto|cpu|cuda|mps`
- `--speaker-output-mode context|json-text`
- `--speaker-unknown-name` (json-text mode only)
- `--speaker-include-score` / `--no-speaker-include-score`

If `--embeddings-file` is omitted, the server automatically uses `<first data dir>/speakers.pkl` when it exists.

Unknown speakers default to `null`.

### Output contracts

- **`context` (default, Home Assistant safe)**
  - `Transcript.text`: plain transcript
  - `Transcript.context.speaker`: `{ "name": "alice", "score": 0.82 }`

- **`json-text`**
  - `Transcript.text`: `{"text":"turn on the light","speaker":{"name":"alice","score":0.82}}`

For Home Assistant Voice Pipeline compatibility, keep `--speaker-output-mode context`.

## Quick Web UI Testing

Start the optional web UI together with the Wyoming server:

```sh
script/run \
  --model tiny-int8 \
  --language en \
  --uri 'tcp://0.0.0.0:10300' \
  --data-dir /data \
  --download-dir /data \
  --web-server
```

Then open `http://localhost:8099`.

The page lets you:

1. upload **or record** speaker sample clips (browser microphone),
2. enroll/update embeddings,
3. upload **or record** a test clip and see transcription + speaker result,
4. inspect the Wyoming-style payload (`context` / `json-text`).

Useful web options:

- `--web-server`
- `--web-host` (default: `0.0.0.0`)
- `--web-port` (default: `8099`)
- `--speaker-reference-dir` (default: `<first data dir>/speaker_samples`)

## NVIDIA GPU / CUDA and Parakeet

For NVIDIA GPUs, use CUDA-enabled torch and the NeMo backend:

```sh
script/run \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --stt-library nemo \
  --device cuda \
  --language en \
  --uri 'tcp://0.0.0.0:10300' \
  --data-dir /data \
  --download-dir /data
```

### CUDA-capable Docker build

Build the image with CUDA torch and include the `nemo` extra:

```sh
docker build . -t wyoming-faster-whisper:cuda \
  --build-arg TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  --build-arg TORCH_PACKAGE='torch==2.5.1+cu121' \
  --build-arg INSTALL_EXTRAS='zeroconf,transformers,sherpa,onnx-asr,speaker,web,nemo'
```

If your host driver only supports CUDA 12.2, keep the `cu121` torch build above and avoid newer CUDA wheels.

Run it with the NVIDIA runtime:

```sh
docker run --rm --gpus all \
  -p 10300:10300 \
  -p 8099:8099 \
  -v /path/to/local/data:/data \
  wyoming-faster-whisper:cuda \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --stt-library nemo \
  --device cuda \
  --language en \
  --web-server
```

## Docker Image

``` sh
docker run --rm \
    -p 10300:10300 \
    -p 8099:8099 \
    -v /path/to/local/data:/data \
    rhasspy/wyoming-whisper \
    --model tiny-int8 --language en --web-server
```

**NOTE**: Models are downloaded to `/data`, so make sure this points to a Docker volume.

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/whisper)
