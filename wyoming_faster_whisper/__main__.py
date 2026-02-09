#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import faster_whisper
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .const import AUTO_LANGUAGE, AUTO_MODEL, PARAKEET_LANGUAGES, SttLibrary
from .dispatch_handler import DispatchEventHandler
from .models import ModelLoader
from .speaker_identifier import (
    SpeakerIdentifier,
    ensure_speaker_dependencies,
    load_embeddings,
)

_LOGGER = logging.getLogger(__name__)


def _torch_runtime_summary() -> str:
    """Summarize torch/cuda runtime state for startup logs."""
    try:
        import torch
    except Exception as err:  # pylint: disable=broad-except
        return f"torch unavailable ({err.__class__.__name__})"

    summary_parts = [f"torch={torch.__version__}"]

    cuda_available = bool(torch.cuda.is_available())
    summary_parts.append(f"cuda_available={cuda_available}")

    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        summary_parts.append(f"cuda_version={cuda_version}")

    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
        except Exception:  # pylint: disable=broad-except
            device_count = 0

        summary_parts.append(f"cuda_devices={device_count}")

        if device_count > 0:
            try:
                device_names = [
                    torch.cuda.get_device_name(index) for index in range(device_count)
                ]
                summary_parts.append(f"gpus={'; '.join(device_names)}")
            except Exception:  # pylint: disable=broad-except
                pass

    return ", ".join(summary_parts)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="faster-whisper",
        help="Enable discovery over zeroconf with optional name (default: faster-whisper)",
    )
    #
    parser.add_argument(
        "--model", default=AUTO_MODEL, help=f"Name of model to use (or {AUTO_MODEL})"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument(
        "--language",
        default=AUTO_LANGUAGE,
        help=f"Default language to set for transcription (default: {AUTO_LANGUAGE})",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=0,
        help="Size of beam during decoding (0 for auto)",
    )
    parser.add_argument(
        "--cpu-threads",
        default=4,
        type=int,
        help="Number of CPU threads to use for inference (default: 4, faster-whisper ony)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window (faster-whisper only)",
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable Silero VAD to filter out non-speech which can reduce hallucinations (default: false, faster-whisper only)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD speech probability threshold (default: 0.5, faster-whisper only)",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=250,
        help="VAD minimum speech duration in ms (default: 250, faster-whisper only)",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=2000,
        help="VAD minimum silence duration in ms to split (default: 2000, faster-whisper only)",
    )
    parser.add_argument(
        "--stt-library",
        choices=[lib.value for lib in SttLibrary],
        default=SttLibrary.AUTO,
        help="Set library to use for speech-to-text (may require extra dependencies)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Don't check HuggingFace hub for updates every time",
    )
    #
    parser.add_argument(
        "--embeddings-file",
        help=(
            "Path to enrolled speaker embeddings pickle file "
            "(default: <first --data-dir>/speakers.pkl)"
        ),
    )
    parser.add_argument(
        "--speaker-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity score to accept a speaker match (default: 0.5)",
    )
    parser.add_argument(
        "--speaker-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for speaker encoder (default: auto)",
    )
    parser.add_argument(
        "--speaker-output-mode",
        choices=["context", "json-text"],
        default="context",
        help="How to return speaker metadata (default: context)",
    )
    parser.add_argument(
        "--speaker-unknown-name",
        help="Override unknown speaker name in json-text mode",
    )
    parser.add_argument(
        "--speaker-include-score",
        dest="speaker_include_score",
        action="store_true",
        default=True,
        help="Include similarity score in speaker metadata (default: true)",
    )
    parser.add_argument(
        "--no-speaker-include-score",
        dest="speaker_include_score",
        action="store_false",
        help="Disable similarity score in speaker metadata",
    )
    parser.add_argument(
        "--speaker-reference-dir",
        help="Directory for speaker reference samples used by the optional web UI",
    )
    parser.add_argument(
        "--web-server",
        action="store_true",
        help="Start optional FastAPI web UI for testing",
    )
    parser.add_argument(
        "--web-host",
        default="0.0.0.0",
        help="Host for optional web UI (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8099,
        help="Port for optional web UI (default: 8099)",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    speaker_identifier: Optional[SpeakerIdentifier] = None
    speaker_output_mode = "context"
    speaker_unknown_name: Optional[str] = None

    default_embeddings_path = Path(args.data_dir[0]) / "speakers.pkl"
    configured_embeddings_path = (
        Path(args.embeddings_file) if args.embeddings_file else default_embeddings_path
    )

    speaker_flags = (
        "--speaker-threshold",
        "--speaker-device",
        "--speaker-output-mode",
        "--speaker-unknown-name",
        "--speaker-include-score",
        "--no-speaker-include-score",
    )
    speaker_flags_without_embeddings = any(
        any(
            (arg == speaker_flag) or arg.startswith(f"{speaker_flag}=")
            for speaker_flag in speaker_flags
        )
        for arg in sys.argv[1:]
    )

    speaker_runtime_needed = args.web_server or configured_embeddings_path.is_file()
    if speaker_runtime_needed:
        try:
            ensure_speaker_dependencies()
        except Exception as err:  # pylint: disable=broad-except
            raise SystemExit(
                "Speaker runtime dependency check failed: "
                f"{err}. Rebuild/install with speaker dependencies "
                "(setuptools<81 is required)."
            ) from err

    if configured_embeddings_path.is_file():
        try:
            embeddings = load_embeddings(configured_embeddings_path)
        except Exception as err:
            raise SystemExit(
                "Failed to load speaker embeddings file "
                f"'{configured_embeddings_path}': {err}"
            ) from err

        try:
            speaker_identifier = SpeakerIdentifier(
                embeddings=embeddings,
                threshold=args.speaker_threshold,
                device=args.speaker_device,
                include_score=args.speaker_include_score,
            )
        except Exception as err:
            raise SystemExit(f"Failed to initialize speaker identifier: {err}") from err

        speaker_output_mode = args.speaker_output_mode
        speaker_unknown_name = args.speaker_unknown_name

        _LOGGER.info(
            "Speaker ID enabled (embeddings=%s, threshold=%.3f, device=%s, output_mode=%s)",
            configured_embeddings_path,
            args.speaker_threshold,
            args.speaker_device,
            speaker_output_mode,
        )
    else:
        if speaker_flags_without_embeddings and (not args.web_server):
            _LOGGER.warning(
                "Ignoring speaker options because embeddings file was not found: %s",
                configured_embeddings_path,
            )
        elif args.embeddings_file:
            _LOGGER.warning(
                "Embeddings file not found yet: %s (speaker ID disabled until file exists)",
                configured_embeddings_path,
            )
        else:
            _LOGGER.info(
                "Speaker ID disabled (no embeddings file at %s)",
                configured_embeddings_path,
            )

    speaker_reference_dir = (
        Path(args.speaker_reference_dir)
        if args.speaker_reference_dir
        else (Path(args.data_dir[0]) / "speaker_samples")
    )

    args.stt_library = SttLibrary(args.stt_library)

    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)

    if args.beam_size <= 0:
        args.beam_size = 1 if is_arm else 5
        _LOGGER.debug("Beam size automatically selected: %s", args.beam_size)

    # Resolve model name
    model_name = args.model
    model_match = re.match(r"^(tiny|base|small|medium)[.-]int8$", args.model)
    if model_match:
        # Original models re-uploaded to huggingface
        model_size = model_match.group(1)
        model_name = f"{model_size}-int8"
        args.model = f"rhasspy/faster-whisper-{model_name}"

    if args.language == AUTO_LANGUAGE:
        # Whisper does not understand auto
        args.language = None

    if args.model == AUTO_MODEL:
        args.model = None

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="faster-whisper",
                description="Faster Whisper transcription with CTranslate2",
                attribution=Attribution(
                    name="Guillaume Klein",
                    url="https://github.com/guillaumekln/faster-whisper/",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="Systran",
                            url="https://huggingface.co/Systran",
                        ),
                        installed=True,
                        languages=sorted(
                            list(
                                # pylint: disable=protected-access
                                set(faster_whisper.tokenizer._LANGUAGE_CODES).union(
                                    PARAKEET_LANGUAGES
                                )
                            )
                        ),
                        version=faster_whisper.__version__,
                    )
                ],
            )
        ],
    )

    vad_parameters: Optional[Dict[str, Any]] = None
    if args.vad_filter:
        vad_parameters = {
            "threshold": args.vad_threshold,
            "min_speech_duration_ms": args.vad_min_speech_ms,
            "min_silence_duration_ms": args.vad_min_silence_ms,
        }

    loader = ModelLoader(
        preferred_stt_library=args.stt_library,
        preferred_language=args.language,
        download_dir=args.download_dir,
        local_files_only=args.local_files_only,
        model=args.model,
        compute_type=args.compute_type,
        device=args.device,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        initial_prompt=args.initial_prompt,
        vad_parameters=vad_parameters,
    )

    # Load model
    _LOGGER.debug("Pre-loading transcriber")
    transcriber = await loader.load_transcriber()

    selected_backend, selected_model = (
        loader.last_selection
        if loader.last_selection is not None
        else (args.stt_library, args.model or AUTO_MODEL)
    )

    effective_device = getattr(transcriber, "device", args.device)
    _LOGGER.info(
        "Runtime selection: backend=%s model=%s device=%s requested_device=%s language=%s | %s",
        selected_backend.value,
        selected_model,
        effective_device,
        args.device,
        args.language or AUTO_LANGUAGE,
        _torch_runtime_summary(),
    )

    web_app = None
    if args.web_server:
        try:
            from .web_server import WebServerConfig, create_web_app

            web_app = create_web_app(
                loader,
                WebServerConfig(
                    speaker_samples_dir=speaker_reference_dir,
                    embeddings_file=configured_embeddings_path,
                    speaker_threshold=args.speaker_threshold,
                    speaker_device=args.speaker_device,
                    speaker_include_score=args.speaker_include_score,
                    speaker_unknown_name=args.speaker_unknown_name,
                    speaker_output_mode=args.speaker_output_mode,
                ),
            )
        except Exception as err:
            raise SystemExit(f"Failed to initialize web server: {err}") from err

        _LOGGER.info(
            "Web UI enabled at http://%s:%s (samples=%s, embeddings=%s)",
            args.web_host,
            args.web_port,
            speaker_reference_dir,
            configured_embeddings_path,
        )

    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    _LOGGER.info("Ready")

    handler_factory = partial(
        DispatchEventHandler,
        wyoming_info,
        loader,
        speaker_identifier=speaker_identifier,
        speaker_output_mode=speaker_output_mode,
        speaker_unknown_name=speaker_unknown_name,
    )

    tasks = [asyncio.create_task(server.run(handler_factory))]

    if web_app is not None:
        from .web_server import run_web_server

        tasks.append(
            asyncio.create_task(run_web_server(web_app, args.web_host, args.web_port))
        )

    await asyncio.gather(*tasks)


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
