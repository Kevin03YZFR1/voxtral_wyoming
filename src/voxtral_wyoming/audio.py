from __future__ import annotations

"""Audio utilities for preprocessing incoming audio.

This initial version is intentionally minimal to avoid heavy dependencies.
"""

from dataclasses import dataclass
from typing import Optional


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
