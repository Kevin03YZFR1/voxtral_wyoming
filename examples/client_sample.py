from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_audio(source: str) -> bytes:
    """Load audio bytes from a local file path or a remote URL."""
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(source) as resp:  # nosec - user-specified URL to public sample
            return resp.read()
    # Treat as a local file path (absolute or relative)
    return Path(source).expanduser().resolve().read_bytes()


def _maybe_convert_to_pcm16(audio_bytes: bytes, sample_rate: int = 16000) -> tuple[bytes, bool]:
    """
    Try to convert input audio (mp3/other) to raw PCM16 mono at the given sample_rate using ffmpeg.

    Returns (converted_bytes, used_ffmpeg).
    If ffmpeg is not available or conversion fails, returns original bytes and False.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return audio_bytes, False

    try:
        proc = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "-",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return proc.stdout, True
    except Exception:
        return audio_bytes, False


def transcribe_sample(
    host: str = "127.0.0.1",
    port: int = int(os.getenv("PORT", "10300")),
    source: str = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
    sample_rate: int = int(os.getenv("SAMPLE_RATE_FALLBACK", "16000")),
    convert: bool = True,
) -> dict:
    """
    Connect to a running voxtral-wyoming server using the Wyoming protocol and return the transcript.

    *source* can be a remote URL (http/https) or a local file path (absolute or relative).

    Flow: Describe -> Transcribe -> AudioStart -> AudioChunk* -> AudioStop -> Transcript
    """
    # Lazy import to keep CLI startup fast and to avoid import cost if unused
    from wyoming.event import write_event, read_event  # type: ignore
    from wyoming.audio import AudioStart, AudioChunk, AudioStop  # type: ignore
    from wyoming.asr import Transcribe, Transcript  # type: ignore
    from wyoming.info import Describe  # type: ignore

    audio_bytes = _load_audio(source)
    used_ffmpeg = False
    if convert:
        converted, used_ffmpeg = _maybe_convert_to_pcm16(audio_bytes, sample_rate=sample_rate)
        audio_bytes = converted

    # Prepare audio format (expect PCM16 mono)
    width = 2
    channels = 1

    # Connect to Wyoming server and speak protocol
    with socket.create_connection((host, port), timeout=10) as sock:
        # Disable socket timeout for long-running transcription
        try:
            sock.settimeout(None)
        except Exception:
            pass
        # Use buffered file interfaces for wyoming.read_event/write_event
        rfile = sock.makefile("rb")
        wfile = sock.makefile("wb")

        # 1) Ask for info (optional but useful)
        write_event(Describe().event(), wfile)

        # 2) Begin transcription session
        write_event(Transcribe().event(), wfile)

        # 3) Send audio start
        write_event(AudioStart(rate=sample_rate, width=width, channels=channels).event(), wfile)

        # 4) Stream audio chunks
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            write_event(
                AudioChunk(rate=sample_rate, width=width, channels=channels, audio=chunk).event(),
                wfile,
            )

        # 5) Finish audio stream
        write_event(AudioStop().event(), wfile)

        # 6) Read events until we get a Transcript or connection closes
        result_text: Optional[str] = None
        result_language: Optional[str] = None
        while True:
            event = read_event(rfile)
            if event is None:
                break
            if Transcript.is_type(event.type):
                tr = Transcript.from_event(event)
                result_text = tr.text
                result_language = tr.language
                break

        # Close buffered files explicitly
        try:
            wfile.flush()
        except Exception:
            pass
        try:
            rfile.close()
            wfile.close()
        except Exception:
            pass

    # Build response JSON
    resp_json: dict = {}
    if result_text is not None:
        resp_json["text"] = result_text
        if result_language is not None:
            resp_json["language"] = result_language
    else:
        resp_json["error"] = "No transcript received from server"

    # Annotate with client-side metadata
    resp_json.setdefault("_client", {})
    resp_json["_client"].update(
        {
            "source": source,
            "converted_pcm16": bool(convert and used_ffmpeg),
            "sample_rate": sample_rate,
            "host": host,
            "port": port,
        }
    )

    return resp_json


def main(argv: Optional[list[str]] = None) -> int:
    # Parse command line argument for env file
    env_file = ".env"
    env_file_explicitly_specified = False
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) > 0:
        env_file = argv[0]
        env_file_explicitly_specified = True

    # Load environment variables from the specified file
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
    elif env_file_explicitly_specified:
        print(f"Error: Environment file {env_file} not found, using system environment variables only", file=sys.stderr)

    # Read configuration from environment variables
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "10300"))
    source = os.getenv("SAMPLE_FILE", "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3")
    sample_rate = int(os.getenv("SAMPLE_RATE_FALLBACK", "16000"))
    convert = os.getenv("CONVERT_AUDIO", "true").lower() in ("true", "1", "yes")

    try:
        result = transcribe_sample(
            host=host,
            port=port,
            source=source,
            sample_rate=sample_rate,
            convert=convert,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Print the transcript text if available, else full JSON
    text = result.get("text")
    if text is not None:
        print(text)
    else:
        print(json.dumps(result, ensure_ascii=False))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
