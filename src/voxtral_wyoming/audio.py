from __future__ import annotations

"""Audio utilities for preprocessing incoming audio.

This initial version is intentionally minimal to avoid heavy dependencies.
"""

import re
import wave
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AudioSpec:
    sample_rate: int = 16000
    channels: int = 1
    sample_width_bytes: int = 2  # PCM16
    endian: str = "little"


def expected_bytes_per_second(spec: AudioSpec) -> int:
    return spec.sample_rate * spec.channels * spec.sample_width_bytes


def clamp_audio_size(audio: bytes, spec: AudioSpec, max_seconds: int = 60) -> bytes:
    """Clamp audio to a maximum number of seconds to avoid unbounded memory use."""
    max_bytes = expected_bytes_per_second(spec) * max_seconds
    if len(audio) > max_bytes:
        return audio[:max_bytes]
    return audio


def pcm16_duration_seconds(length_bytes: int, sample_rate: int, channels: int = 1) -> float:
    """Compute duration in seconds for PCM16 audio by length.

    Args:
        length_bytes: Number of bytes in the PCM16 buffer.
        sample_rate: Samples per second.
        channels: Number of channels (default 1 = mono).
    """
    if sample_rate <= 0 or channels <= 0:
        return 0.0
    bytes_per_second = sample_rate * channels * 2  # PCM16 = 2 bytes per sample
    if bytes_per_second == 0:
        return 0.0
    return length_bytes / bytes_per_second


def _sanitize_text_for_filename(text: str, max_length: int = 100) -> str:
    """Sanitize text for use in filenames.

    Args:
        text: The text to sanitize.
        max_length: Maximum length of the sanitized text.

    Returns:
        Sanitized text safe for use in filenames.
    """
    if not text:
        return ""

    # Take only the first max_length characters
    text = text[:max_length]

    # Replace path separators and other problematic characters
    # Replace with underscore: / \ : * ? " < > | and newlines/tabs
    text = re.sub(r'[/\\:*?"<>|\r\n\t]', '_', text)

    # Replace multiple spaces or underscores with a single underscore
    text = re.sub(r'[\s_]+', '_', text)

    # Remove leading/trailing underscores and spaces
    text = text.strip('_ ')

    return text


def save_audio_as_wav(
    audio_pcm: bytes,
    sample_rate: int,
    output_dir: str | Path,
    channels: int = 1,
    sample_width: int = 2,
    text: str = "",
) -> Path:
    """Save PCM16 audio data as a WAV file with timestamp-based naming.

    Args:
        audio_pcm: Raw PCM16 audio bytes (little-endian).
        sample_rate: Sample rate in Hz.
        output_dir: Directory where the WAV file will be saved.
        channels: Number of audio channels (default 1 = mono).
        sample_width: Sample width in bytes (default 2 = 16-bit).
        text: Optional transcribed text to include in filename (first 100 chars).

    Returns:
        Path to the created WAV file.

    Raises:
        OSError: If directory creation or file writing fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Add sanitized text to filename if provided
    sanitized_text = _sanitize_text_for_filename(text, max_length=100)
    if sanitized_text:
        filename = f"audio_{timestamp}_{sanitized_text}.wav"
    else:
        filename = f"audio_{timestamp}.wav"

    filepath = output_path / filename

    # Write WAV file
    with wave.open(str(filepath), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_pcm)

    return filepath
