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
    port: int = int(os.getenv("WYOMING_PORT", "10300")),
    url: str = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
    convert: bool = True,
) -> dict:
    """
    Connects to a running voxtral-wyoming server, sends audio bytes, and returns the JSON response.

    By default downloads the Obama sample MP3, attempts conversion to PCM16 (if ffmpeg is available),
    and then streams the bytes to the server's stub protocol.
    """
    audio_bytes = _download(url)
    used_ffmpeg = False
    if convert:
        converted, used_ffmpeg = _maybe_convert_to_pcm16(audio_bytes, sample_rate=sample_rate)
        audio_bytes = converted

    # Connect to server and send bytes
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(audio_bytes)
        try:
            sock.shutdown(socket.SHUT_WR)
        except Exception:
            pass
        chunks = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)

    resp_raw = b"".join(chunks).strip()
    # Allow servers to respond with JSON + newline
    if b"\n" in resp_raw:
        resp_raw = resp_raw.split(b"\n", 1)[0]

    try:
        resp_json = json.loads(resp_raw.decode("utf-8"))
    except Exception:
        # Fallback to plain text result if server didn't send JSON
        resp_json = {"raw": resp_raw.decode("utf-8", errors="replace")}

    # Annotate with client-side metadata
    resp_json.setdefault("_client", {})
    resp_json["_client"].update({
        "url": url,
        "converted_pcm16": bool(convert and used_ffmpeg),
        "sample_rate": sample_rate,
        "host": host,
        "port": port,
    })

    return resp_json


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Send sample audio to voxtral-wyoming server and print transcript")
    parser.add_argument("--host", default=os.getenv("WYOMING_HOST", "127.0.0.1"), help="Server host (default: 127.0.0.1 or $WYOMING_HOST)")
    parser.add_argument("--port", type=int, default=int(os.getenv("WYOMING_PORT", "10300")), help="Server port (default: 10300 or $WYOMING_PORT)")
    parser.add_argument("--url", default="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3", help="Audio file URL to transcribe")
    parser.add_argument("--sample-rate", type=int, default=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")), help="Target sample rate for PCM16 conversion")
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
