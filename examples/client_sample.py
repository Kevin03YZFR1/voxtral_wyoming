from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import shutil
import urllib.request
from typing import Optional


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:  # nosec - user-specified URL to public sample
        return resp.read()


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
    url: str = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000")),
    convert: bool = True,
) -> dict:
    """
    Connect to a running voxtral-wyoming server using the Wyoming protocol and return the transcript.

    Flow: Describe -> Transcribe -> AudioStart -> AudioChunk* -> AudioStop -> Transcript
    """
    # Lazy import to keep CLI startup fast and to avoid import cost if unused
    from wyoming.event import write_event, read_event  # type: ignore
    from wyoming.audio import AudioStart, AudioChunk, AudioStop  # type: ignore
    from wyoming.asr import Transcribe, Transcript  # type: ignore
    from wyoming.info import Describe  # type: ignore

    audio_bytes = _download(url)
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
            "url": url,
            "converted_pcm16": bool(convert and used_ffmpeg),
            "sample_rate": sample_rate,
            "host": host,
            "port": port,
        }
    )

    return resp_json


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Send sample audio to voxtral-wyoming server and print transcript")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="Server host (default: 127.0.0.1 or $HOST)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "10300")), help="Server port (default: 10300 or $PORT)")
    parser.add_argument("--url", default="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3", help="Audio file URL to transcribe")
    parser.add_argument("--sample-rate", type=int, default=int(os.getenv("SAMPLE_RATE", "16000")), help="Target sample rate for PCM16 conversion")
    parser.add_argument("--no-convert", action="store_true", help="Do not attempt conversion with ffmpeg; send bytes as-is")

    args = parser.parse_args(argv)

    try:
        result = transcribe_sample(
            host=args.host,
            port=args.port,
            url=args.url,
            sample_rate=args.sample_rate,
            convert=not args.no_convert,
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
