from __future__ import annotations

"""Voxtral-backed transcriber implementation (placeholder).

This module scaffolds the interface for a local Voxtral model runner. The actual
loading and inference logic will be added once the supported runtime and model
format are finalized for offline use.

Environment variables (proposed):
- VOXTRAL_MODEL_DIR / VOXTRAL_MODEL_PATH
- VOXTRAL_DEVICE: cpu|cuda|mps
- VOXTRAL_DTYPE: fp32|fp16|bf16
- VOXTRAL_LANGUAGE: locale like en-US

The implementation MUST remain offline-only.
"""

import os
from dataclasses import dataclass
from typing import Optional

from .base import ITranscriber, TranscriptionResult


@dataclass
class VoxtralConfig:
    model_path: Optional[str] = os.getenv("VOXTRAL_MODEL_PATH") or os.getenv("VOXTRAL_MODEL_DIR")
    device: str = os.getenv("VOXTRAL_DEVICE", "cpu")
    dtype: str = os.getenv("VOXTRAL_DTYPE", "fp32")
    language: Optional[str] = os.getenv("VOXTRAL_LANGUAGE")


class VoxtralTranscriber(ITranscriber):
    """Placeholder Voxtral transcriber.

    For now, this simply raises NotImplementedError to signal that the model
    runtime is not yet wired. It documents the intended configuration surface.
    """

    def __init__(self, config: Optional[VoxtralConfig] = None):
        self.config = config or VoxtralConfig()

    def transcribe(self, audio_pcm: bytes, sample_rate: int, language: Optional[str] = None) -> TranscriptionResult:
        raise NotImplementedError(
            "VoxtralTranscriber is not yet implemented. This placeholder documents the intended interface."
        )
