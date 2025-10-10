from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

import click

from .transcriber.dummy import DummyTranscriber
from .transcriber.voxtral import VoxtralTranscriber, VoxtralConfig
from .transcriber.base import ITranscriber
from .audio import AudioSpec, clamp_audio_size

# Environment variable defaults
DEFAULT_HOST = os.getenv("WYOMING_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("WYOMING_PORT", "10300"))
DEFAULT_LANGUAGE = os.getenv("VOXTRAL_LANGUAGE", "en-US")
DEFAULT_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_PROTOCOL = os.getenv("WYOMING_PROTOCOL", "wyoming").lower()  # wyoming|stub
DEFAULT_BACKEND = os.getenv("VOXTRAL_BACKEND", "dummy").lower()  # dummy|voxtral
DEFAULT_MAX_SECONDS = int(os.getenv("AUDIO_MAX_SECONDS", "60"))

_LOGGER = logging.getLogger("voxtral_wyoming")


async def _stub_handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    language: str,
    sample_rate: int,
    transcriber: ITranscriber,
    max_seconds: int,
) -> None:
    """A simple TCP handler that reads all incoming bytes and returns a JSON transcript.

    This is a temporary placeholder until full Wyoming protocol wiring is implemented.
    """
    addr = writer.get_extra_info("peername")
    _LOGGER.info("Client connected from %s", addr)

    # Read all data until client closes the connection
    try:
        data = await reader.read()  # read until EOF
    except Exception as e:  # pragma: no cover - defensive
        _LOGGER.exception("Error reading from client: %s", e)
        data = b""

    # Clamp audio to avoid unbounded memory usage
    spec = AudioSpec(sample_rate=sample_rate)
    data = clamp_audio_size(data, spec, max_seconds=max_seconds)

    result = transcriber.transcribe(data, sample_rate=sample_rate, language=language)

    response = {
        "service": "voxtral-wyoming-stub",
        "protocol": "stub-json",  # Not Wyoming yet
        "language": result.language,
        "text": result.text,
    }

    try:
        writer.write(json.dumps(response).encode("utf-8") + b"\n")
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()
        _LOGGER.info("Client disconnected: %s", addr)


async def _run_stub_server(host: str, port: int, language: str, sample_rate: int, transcriber: ITranscriber, max_seconds: int) -> None:
    server = await asyncio.start_server(
        lambda r, w: _stub_handle_client(
            r,
            w,
            language=language,
            sample_rate=sample_rate,
            transcriber=transcriber,
            max_seconds=max_seconds,
        ),
        host,
        port,
    )

    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    _LOGGER.info("Stub server listening on %s", addrs)

    async with server:
        await server.serve_forever()


async def _run_wyoming_server(host: str, port: int, language: str, sample_rate: int, transcriber: ITranscriber, max_seconds: int) -> None:
    """Attempt to run a Wyoming protocol server.

    For now, this is a thin wrapper that validates the `wyoming` package is importable
    and falls back to the stub JSON server until full protocol support is implemented.
    """
    try:
        import wyoming  # type: ignore

        _ver = getattr(wyoming, "__version__", "unknown")
        _LOGGER.info("Wyoming package detected (version=%s). Wyoming protocol handling is not yet implemented; falling back to stub server for now.", _ver)
    except Exception as e:  # pragma: no cover - environment-dependent
        _LOGGER.warning("Wyoming package not available (%s). Falling back to stub server.", e)

    await _run_stub_server(host, port, language, sample_rate, transcriber, max_seconds)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--host", envvar="WYOMING_HOST", default=DEFAULT_HOST, show_default=True, help="Bind host")
@click.option("--port", envvar="WYOMING_PORT", default=DEFAULT_PORT, type=int, show_default=True, help="Bind port")
@click.option(
    "--language",
    envvar="VOXTRAL_LANGUAGE",
    default=DEFAULT_LANGUAGE,
    show_default=True,
    help="Language/locale hint (e.g., en-US)",
)
@click.option(
    "--sample-rate",
    envvar="AUDIO_SAMPLE_RATE",
    default=DEFAULT_SAMPLE_RATE,
    type=int,
    show_default=True,
    help="Expected audio sample rate (Hz)",
)
@click.option(
    "--protocol",
    envvar="WYOMING_PROTOCOL",
    default=DEFAULT_PROTOCOL,
    type=click.Choice(["wyoming", "stub"], case_sensitive=False),
    show_default=True,
    help="Protocol to serve (wyoming is recommended; falls back to stub behavior until implemented)",
)
@click.option(
    "--backend",
    envvar="VOXTRAL_BACKEND",
    default=DEFAULT_BACKEND,
    type=click.Choice(["dummy", "voxtral"], case_sensitive=False),
    show_default=True,
    help="Transcription backend to use (voxtral not implemented yet)",
)
@click.option(
    "--max-seconds",
    envvar="AUDIO_MAX_SECONDS",
    default=DEFAULT_MAX_SECONDS,
    type=int,
    show_default=True,
    help="Clamp incoming audio to this many seconds (safety)",
)
@click.option(
    "--log-level",
    envvar="LOG_LEVEL",
    default=DEFAULT_LOG_LEVEL,
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Logging level",
)
def cli(host: str, port: int, language: str, sample_rate: int, protocol: str, backend: str, max_seconds: int, log_level: str) -> None:
    """Start the Voxtral Wyoming STT service.

    Currently supports a stub JSON server. The `wyoming` protocol option is accepted
    and will fall back to stub behavior until full Wyoming support is implemented.
    """
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _LOGGER.info(
        "Starting Voxtral Wyoming STT on %s:%d | language=%s sample_rate=%d protocol=%s backend=%s max_seconds=%d",
        host,
        port,
        language,
        sample_rate,
        protocol,
        backend,
        max_seconds,
    )

    # Select backend
    backend_lower = backend.lower()
    transcriber: ITranscriber
    if backend_lower == "voxtral":
        _LOGGER.warning("Voxtral backend selected but not yet implemented; falling back to dummy transcriber")
        transcriber = DummyTranscriber(text="Hello from Voxtral Wyoming stub", language=language)
        # In a future iteration, replace with: VoxtralTranscriber(VoxtralConfig())
    else:
        transcriber = DummyTranscriber(text="Hello from Voxtral Wyoming stub", language=language)

    try:
        if protocol.lower() == "wyoming":
            asyncio.run(_run_wyoming_server(host, port, language, sample_rate, transcriber, max_seconds))
        else:
            asyncio.run(_run_stub_server(host, port, language, sample_rate, transcriber, max_seconds))
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down (keyboard interrupt)")


if __name__ == "__main__":  # pragma: no cover
    cli()
