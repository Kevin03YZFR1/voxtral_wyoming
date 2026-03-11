from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .transcriber.voxtral import VoxtralTranscriber, VoxtralConfig
from .transcriber.base import ITranscriber
from .audio import AudioSpec, clamp_audio_size, save_audio_as_wav

_LOGGER = logging.getLogger("voxtral_wyoming")


async def _wyoming_handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    language: str,
    default_sample_rate: int,
    transcriber: ITranscriber,
    max_seconds: float,
    save_audio: bool,
    audio_save_dir: str,
    model_ready: asyncio.Event,
    model_error: list,
) -> None:
    """Handle a single Wyoming TCP connection for ASR.

    Expects Describe? → Transcribe? → AudioStart → AudioChunk* → AudioStop,
    and responds with Info (optional) and a final Transcript.
    """
    addr = writer.get_extra_info("peername")
    _LOGGER.debug("Client connected from %s", addr)

    # Wait for the model to finish loading before processing any events.
    # The TCP connection is accepted immediately so clients don't get
    # "connection refused" during startup — they just wait briefly.
    if not model_ready.is_set():
        _LOGGER.info("Client %s connected while model is still loading, waiting...", addr)
        await model_ready.wait()

    if model_error:
        _LOGGER.error("Model failed to load, rejecting client %s", addr)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        return

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
                    languages=transcriber.supported_languages,
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
                    result = transcriber.transcribe(audio_pcm, sample_rate=sample_rate, locale=lang_hint)
                    text = result.text or ""
                    lang_out = result.language or lang_hint
                except Exception as e:
                    _LOGGER.exception("Transcription failed: %s", e)
                    text = ""
                    lang_out = lang_hint

                # Save audio if enabled (after transcription to include text in filename)
                if save_audio and audio_pcm:
                    try:
                        saved_path = save_audio_as_wav(
                            audio_pcm,
                            sample_rate=sample_rate,
                            output_dir=audio_save_dir,
                            text=text,
                        )
                        _LOGGER.info("Saved audio to %s", saved_path)
                    except Exception as e:
                        _LOGGER.error("Failed to save audio: %s", e, exc_info=True)
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


async def _run_wyoming_server(host: str, port: int, language: str, sample_rate: int, transcriber: ITranscriber, max_seconds: float, save_audio: bool, audio_save_dir: str) -> None:
    """Run a Wyoming TCP server over asyncio that handles ASR streams.

    The server starts accepting connections immediately. If the model is
    still loading, client handlers wait for the ``model_ready`` event
    before processing — so clients get a brief pause instead of a
    connection-refused error.
    """
    model_ready = asyncio.Event()
    model_error: list[Exception] = []

    async def _load_model() -> None:
        """Load the model in a background thread so the event loop stays free."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, transcriber._ensure_loaded)
            _LOGGER.info("Model loaded and ready for requests")
        except Exception as e:
            _LOGGER.exception("Failed to load model in background: %s", e)
            model_error.append(e)
        finally:
            model_ready.set()

    # Kick off model loading in the background
    asyncio.create_task(_load_model())

    server = await asyncio.start_server(
        lambda r, w: _wyoming_handle_client(
            r,
            w,
            language=language,
            default_sample_rate=sample_rate,
            transcriber=transcriber,
            max_seconds=max_seconds,
            save_audio=save_audio,
            audio_save_dir=audio_save_dir,
            model_ready=model_ready,
            model_error=model_error,
        ),
        host,
        port,
    )

    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    _LOGGER.info("Wyoming server listening on %s (model loading in background...)", addrs)

    async with server:
        await server.serve_forever()


def cli() -> None:
    """Start the Voxtral Wyoming STT service using the Wyoming protocol and Voxtral backend."""
    # Parse command line argument for env file
    env_file = ".env"
    env_file_explicitly_specified = False
    if len(sys.argv) > 1:
        env_file = sys.argv[1]
        env_file_explicitly_specified = True

    # Load environment variables from the specified file
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
        _LOGGER.info(f"Loaded environment variables from {env_file}")
    else:
        if env_file_explicitly_specified:
            _LOGGER.error(f"Environment file {env_file} not found, using system environment variables only")
        else:
            _LOGGER.warning(f"Environment file {env_file} not found, using system environment variables only")

    # Read configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10300"))
    language = os.getenv("LANGUAGE_FALLBACK", "en-US")
    sample_rate = int(os.getenv("SAMPLE_RATE_FALLBACK", "16000"))
    max_seconds = float(os.getenv("MAX_SECONDS", "30"))
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    save_audio = os.getenv("SAVE_AUDIO", "false").lower() in ("true", "1", "yes")
    audio_save_dir = os.getenv("AUDIO_SAVE_DIR", "/output/audio")

    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _LOGGER.info(
        "Starting Voxtral Wyoming STT on %s:%d | language=%s sample_rate=%d max_seconds=%.1f",
        host,
        port,
        language,
        sample_rate,
        max_seconds,
    )

    if save_audio:
        _LOGGER.info("Audio saving enabled: files will be saved to %s", audio_save_dir)
    else:
        _LOGGER.info("Audio saving disabled")

    # Create transcriber with deferred model loading.
    # The heavy model weights are loaded asynchronously once the server
    # is already listening, so early client connections wait instead of
    # being refused.
    transcriber: ITranscriber = VoxtralTranscriber(VoxtralConfig(), eager=False)

    try:
        asyncio.run(_run_wyoming_server(host, port, language, sample_rate, transcriber, max_seconds, save_audio, audio_save_dir))
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down (keyboard interrupt)")


if __name__ == "__main__":  # pragma: no cover
    cli()
