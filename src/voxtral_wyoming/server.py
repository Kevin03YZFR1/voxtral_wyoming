from __future__ import annotations

import asyncio
import logging
import os
import time

import click

from .transcriber.voxtral import VoxtralTranscriber, VoxtralConfig
from .transcriber.base import ITranscriber
from .audio import AudioSpec, clamp_audio_size

# Environment variable defaults
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", "10300"))
DEFAULT_LANGUAGE = os.getenv("LANGUAGE", "en-US")
DEFAULT_SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_MAX_SECONDS = int(os.getenv("MAX_SECONDS", "60"))

_LOGGER = logging.getLogger("voxtral_wyoming")


async def _wyoming_handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    language: str,
    default_sample_rate: int,
    transcriber: ITranscriber,
    max_seconds: int,
) -> None:
    """Handle a single Wyoming TCP connection for ASR.

    Expects Describe? → Transcribe? → AudioStart → AudioChunk* → AudioStop,
    and responds with Info (optional) and a final Transcript.
    """
    addr = writer.get_extra_info("peername")
    _LOGGER.debug("Client connected from %s", addr)

    try:
        # Lazy imports to keep module import cheap
        from wyoming.event import async_read_event, async_write_event  # type: ignore
        from wyoming.audio import AudioStart, AudioChunk, AudioStop  # type: ignore
        from wyoming.asr import Transcribe, Transcript  # type: ignore
        from wyoming.info import Describe, Info, AsrProgram, AsrModel, Attribution  # type: ignore
        from voxtral_wyoming import __version__ as VW_VERSION  # local version
    except Exception as e:  # pragma: no cover - environment dependent
        _LOGGER.exception("Wyoming package not available or incompatible: %s", e)
        try:
            writer.close()
            await writer.wait_closed()
        finally:
            return

    audio = bytearray()
    sample_rate = default_sample_rate
    lang_hint = language

    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                break

            if Describe.is_type(event.type):
                _LOGGER.debug("Received Describe event from client %s", addr)

                attribution = Attribution(
                    name="Voxtral Wyoming",
                    url="https://github.com/Johnson145/voxtral_wyoming",
                )
                asr_model = AsrModel(
                    name="voxtral",
                    attribution=attribution,
                    installed=True,
                    description="Offline STT with Mistral Voxtral",
                    version=VW_VERSION,
                    languages=[lang_hint],
                )
                asr_program = AsrProgram(
                    name="voxtral-wyoming",
                    attribution=attribution,
                    installed=True,
                    description="Wyoming-compatible STT service",
                    version=VW_VERSION,
                    models=[asr_model],
                    supports_transcript_streaming=False,
                )
                try:
                    await async_write_event(Info(asr=[asr_program]).event(), writer)
                except (ConnectionResetError, BrokenPipeError, OSError):
                    _LOGGER.warning("Client disconnected during Info write: %s", addr)
                    break

            elif Transcribe.is_type(event.type):
                transcribe = Transcribe.from_event(event)

                log_parts = [
                    f"model: {transcribe.name if transcribe.name else 'default'}",
                    f"language: {transcribe.language if transcribe.language else 'default'}",
                ]
                if transcribe.context:
                    log_parts.append(f"context: {transcribe.context}")
                _LOGGER.debug(
                    "Received Transcribe event from client %s (%s)",
                    addr,
                    ", ".join(log_parts)
                )

                if transcribe.language:
                    lang_hint = transcribe.language

            elif AudioStart.is_type(event.type):
                audio_start = AudioStart.from_event(event)
                # Note: We expect width=2, channels=1 (PCM16 mono)

                # Prefer sample rate from client if provided
                sample_rate = getattr(audio_start, "rate", sample_rate) or sample_rate

                log_msg = f"Received AudioStart event from client {addr} (rate: {sample_rate} Hz, width: {getattr(audio_start, 'width', 'unknown')}, channels: {getattr(audio_start, 'channels', 'unknown')}"
                if audio_start.timestamp is not None:
                    log_msg += f", timestamp: {audio_start.timestamp}ms"
                log_msg += ")"
                _LOGGER.debug(log_msg)

            elif AudioChunk.is_type(event.type):
                audio_chunk = AudioChunk.from_event(event)
                if audio_chunk.audio:
                    # Too verbose for permanent logging
                    # log_msg = f"Received AudioChunk event from client {addr} (chunk size: {len(audio_chunk.audio)} bytes, total accumulated: {len(audio) + len(audio_chunk.audio)} bytes"
                    # if audio_chunk.timestamp is not None:
                    #     log_msg += f", timestamp: {audio_chunk.timestamp}ms"
                    # log_msg += ")"
                    # _LOGGER.debug(log_msg)

                    audio.extend(audio_chunk.audio)

            elif AudioStop.is_type(event.type):
                audio_stop = AudioStop.from_event(event)

                log_msg = f"Received AudioStop event from client {addr} (total audio received: {len(audio)} bytes"
                if audio_stop.timestamp is not None:
                    log_msg += f", timestamp: {audio_stop.timestamp}ms"
                log_msg += ")"
                _LOGGER.debug(log_msg)

                # Start timing for overall request processing
                request_start = time.perf_counter()

                # Clamp for safety and transcribe
                spec = AudioSpec(sample_rate=sample_rate)
                audio_pcm = clamp_audio_size(bytes(audio), spec, max_seconds=max_seconds)

                try:
                    result = transcriber.transcribe(audio_pcm, sample_rate=sample_rate, language=lang_hint)
                    text = result.text or ""
                    lang_out = result.language or lang_hint
                except Exception as e:
                    _LOGGER.exception("Transcription failed: %s", e)
                    text = ""
                    lang_out = lang_hint
                try:
                    await async_write_event(Transcript(text=text, language=lang_out).event(), writer)

                    # Log overall request processing time
                    request_time = time.perf_counter() - request_start
                    _LOGGER.debug(f"Request processing completed in {request_time:.2f}s (client: {addr})")
                except (ConnectionResetError, BrokenPipeError, OSError):
                    _LOGGER.warning("Client disconnected before receiving Transcript: %s", addr)
                break

            else:
                # Ignore other messages
                _LOGGER.warning(
                    "Received unknown/unhandled event from client %s (event type: %s)",
                    addr,
                    event.type if hasattr(event, 'type') else 'unknown'
                )
                continue
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        _LOGGER.debug("Client disconnected: %s", addr)


async def _run_wyoming_server(host: str, port: int, language: str, sample_rate: int, transcriber: ITranscriber, max_seconds: int) -> None:
    """Run a Wyoming TCP server over asyncio that handles ASR streams."""
    server = await asyncio.start_server(
        lambda r, w: _wyoming_handle_client(
            r,
            w,
            language=language,
            default_sample_rate=sample_rate,
            transcriber=transcriber,
            max_seconds=max_seconds,
        ),
        host,
        port,
    )

    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    _LOGGER.info("Wyoming server successfully started. Ready to listen for client calls on %s.", addrs)

    async with server:
        await server.serve_forever()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--host", envvar="HOST", default=DEFAULT_HOST, show_default=True, help="Bind host")
@click.option("--port", envvar="PORT", default=DEFAULT_PORT, type=int, show_default=True, help="Bind port")
@click.option(
    "--language",
    envvar="LANGUAGE",
    default=DEFAULT_LANGUAGE,
    show_default=True,
    help="Language/locale hint (e.g., en-US)",
)
@click.option(
    "--sample-rate",
    envvar="SAMPLE_RATE",
    default=DEFAULT_SAMPLE_RATE,
    type=int,
    show_default=True,
    help="Expected audio sample rate (Hz)",
)
@click.option(
    "--max-seconds",
    envvar="MAX_SECONDS",
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
        _LOGGER.info("Voxtral transcriber initialization completed")
    except Exception as e:
        _LOGGER.exception("Failed to initialize Voxtral backend: %s", e)
        raise SystemExit(2)

    try:
        asyncio.run(_run_wyoming_server(host, port, language, sample_rate, transcriber, max_seconds))
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down (keyboard interrupt)")


if __name__ == "__main__":  # pragma: no cover
    cli()
