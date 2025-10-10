from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import click

from .transcriber.voxtral import VoxtralTranscriber, VoxtralConfig
from .transcriber.base import ITranscriber
from .audio import AudioSpec, clamp_audio_size

# Environment variable defaults
DEFAULT_HOST = os.getenv("WYOMING_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("WYOMING_PORT", "10300"))
DEFAULT_LANGUAGE = os.getenv("VOXTRAL_LANGUAGE", "en-US")
DEFAULT_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_MAX_SECONDS = int(os.getenv("AUDIO_MAX_SECONDS", "60"))

_LOGGER = logging.getLogger("voxtral_wyoming")


async def _wyoming_handle_stream(
    stream,
    *,
    language: str,
    default_sample_rate: int,
    transcriber: ITranscriber,
    max_seconds: int,
) -> None:
    """Handle a single Wyoming stream for ASR.

    Expects an AudioStart → zero or more AudioChunk → AudioStop sequence,
    optionally preceded by a Transcribe request. Sends a Transcript response.
    """
    try:
        # Lazy imports to keep module import cheap
        from wyoming.audio import AudioStart, AudioChunk, AudioStop  # type: ignore
        from wyoming.asr import Transcribe, Transcript  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        _LOGGER.exception("Wyoming package not available or incompatible: %s", e)
        await stream.close()
        return

    audio = bytearray()
    sample_rate = default_sample_rate

    while True:
        message = await stream.receive()
        if message is None:
            break

        if isinstance(message, AudioStart):
            # Prefer rate from client if provided
            sample_rate = getattr(message, "rate", sample_rate) or sample_rate
        elif isinstance(message, AudioChunk):
            chunk = getattr(message, "audio", b"")
            if chunk:
                audio.extend(chunk)
        elif isinstance(message, AudioStop):
            # Clamp for safety and transcribe
            spec = AudioSpec(sample_rate=sample_rate)
            audio_pcm = clamp_audio_size(bytes(audio), spec, max_seconds=max_seconds)

            result = transcriber.transcribe(audio_pcm, sample_rate=sample_rate, language=language)
            await stream.send(Transcript(text=result.text or ""))
            await stream.close()
            break
        elif isinstance(message, Transcribe):
            # Trigger received; actual audio will follow
            continue
        else:
            # Ignore other messages
            continue


async def _run_wyoming_server(host: str, port: int, language: str, sample_rate: int, transcriber: ITranscriber, max_seconds: int) -> None:
    """Run a Wyoming TCP server that handles ASR streams."""
    try:
        from wyoming.transport.tcp import TcpServer  # type: ignore
    except Exception as e:  # pragma: no cover - environment-dependent
        _LOGGER.exception("Wyoming TCP transport not available: %s", e)
        raise

    server = TcpServer(host=host, port=port)
    await server.start()

    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    _LOGGER.info("Wyoming server listening on %s", addrs)

    async with server:
        async for stream in server:
            # Handle each stream concurrently
            asyncio.create_task(
                _wyoming_handle_stream(
                    stream,
                    language=language,
                    default_sample_rate=sample_rate,
                    transcriber=transcriber,
                    max_seconds=max_seconds,
                )
            )


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
def cli(host: str, port: int, language: str, sample_rate: int, max_seconds: int, log_level: str) -> None:
    """Start the Voxtral Wyoming STT service using the Wyoming protocol and Voxtral backend."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _LOGGER.info(
        "Starting Voxtral Wyoming STT on %s:%d | language=%s sample_rate=%d max_seconds=%d",
        host,
        port,
        language,
        sample_rate,
        max_seconds,
    )

    # Initialize Voxtral transcriber
    try:
        transcriber: ITranscriber = VoxtralTranscriber(VoxtralConfig())
        _LOGGER.info("Using Voxtral transcriber backend")
    except Exception as e:
        _LOGGER.exception("Failed to initialize Voxtral backend: %s", e)
        raise SystemExit(2)

    try:
        asyncio.run(_run_wyoming_server(host, port, language, sample_rate, transcriber, max_seconds))
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down (keyboard interrupt)")


if __name__ == "__main__":  # pragma: no cover
    cli()
