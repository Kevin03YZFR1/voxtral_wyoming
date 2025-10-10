from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

import click

from .transcriber.dummy import DummyTranscriber

# Environment variable defaults
DEFAULT_HOST = os.getenv("WYOMING_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("WYOMING_PORT", "10300"))
DEFAULT_LANGUAGE = os.getenv("VOXTRAL_LANGUAGE", "en-US")
DEFAULT_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_LOGGER = logging.getLogger("voxtral_wyoming")


async def _stub_handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, language: str, sample_rate: int) -> None:
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

    transcriber = DummyTranscriber(text="Hello from Voxtral Wyoming stub", language=language)
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


async def _run_stub_server(host: str, port: int, language: str, sample_rate: int) -> None:
    server = await asyncio.start_server(
        lambda r, w: _stub_handle_client(r, w, language=language, sample_rate=sample_rate),
        host,
        port,
    )

    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    _LOGGER.info("Stub server listening on %s", addrs)

    async with server:
        await server.serve_forever()


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
    "--log-level",
    envvar="LOG_LEVEL",
    default=DEFAULT_LOG_LEVEL,
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Logging level",
)
def cli(host: str, port: int, language: str, sample_rate: int, log_level: str) -> None:
    """Start the Voxtral Wyoming STT service.

    Note: This initial version runs a stub TCP server returning a fixed transcript.
    Wyoming protocol support will be added next, using the installed `wyoming` package.
    """
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _LOGGER.info(
        "Starting Voxtral Wyoming STT (stub) on %s:%d | language=%s sample_rate=%d",
        host,
        port,
        language,
        sample_rate,
    )

    try:
        asyncio.run(_run_stub_server(host, port, language, sample_rate))
    except KeyboardInterrupt:
        _LOGGER.info("Shutting down (keyboard interrupt)")


if __name__ == "__main__":  # pragma: no cover
    cli()
