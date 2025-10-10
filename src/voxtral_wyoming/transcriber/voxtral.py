from __future__ import annotations

"""Voxtral-backed transcriber implementation using Mistral's Voxtral model.

Offline-only execution that loads local model files (no network calls).

Environment variables:
- VOXTRAL_MODEL_DIR / VOXTRAL_MODEL_PATH: Local path to the Voxtral model dir (recommended)
- VOXTRAL_DEVICE: cpu|cuda|mps (default: cpu)
- VOXTRAL_DTYPE: fp32|fp16|bf16 (default: fp32)
- VOXTRAL_LANGUAGE: locale like en-US (default: None → processor/model default)
- VOXTRAL_MAX_NEW_TOKENS: generation length (default: 500)

Note: If a Hugging Face repo ID is provided instead of a path, loading will still
use local_files_only=True. Ensure the model is present in the local HF cache or
provide VOXTRAL_MODEL_PATH to a directory on disk.
"""

import os
from dataclasses import dataclass
from typing import Optional

from .base import ITranscriber, TranscriptionResult


@dataclass
class VoxtralConfig:
    model_path: Optional[str] = os.getenv("VOXTRAL_MODEL_PATH") or os.getenv("VOXTRAL_MODEL_DIR") or "mistralai/Voxtral-Mini-3B-2507"
    device: str = os.getenv("VOXTRAL_DEVICE", "cpu")
    dtype: str = os.getenv("VOXTRAL_DTYPE", "fp32")
    language: Optional[str] = os.getenv("VOXTRAL_LANGUAGE")
    max_new_tokens: int = int(os.getenv("VOXTRAL_MAX_NEW_TOKENS", "500"))


def _locale_to_lang(locale: Optional[str]) -> Optional[str]:
    if not locale:
        return None
    # Convert 'en-US' or 'en_US' → 'en'
    return locale.split("-")[0].split("_")[0]


def _map_dtype(device: str, dtype_str: str):
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover - only when voxtral backend is used
        raise ImportError(
            "PyTorch is required for VoxtralTranscriber. Install torch >= 2.3."
        ) from e

    norm = dtype_str.lower()
    if device == "cpu":
        # Keep it safe/portable on CPU by default
        return torch.float32
    if norm in ("bf16", "bfloat16"):
        return torch.bfloat16
    if norm in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def _pcm16_le_bytes_to_float32(audio_pcm: bytes):
    # Convert little-endian PCM16 mono to float32 in [-1, 1]
    try:
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover - only when voxtral backend is used
        raise ImportError(
            "NumPy is required for VoxtralTranscriber. Install numpy."
        ) from e

    if not audio_pcm:
        return np.zeros(0, dtype=np.float32)

    arr = np.frombuffer(audio_pcm, dtype='<i2')
    return (arr.astype(np.float32) / 32768.0)


class VoxtralTranscriber(ITranscriber):
    """Local Voxtral transcriber implementation.

    Loads processor and model from local files/cache and runs a single-shot
    transcription for PCM16 mono audio.
    """

    def __init__(self, config: Optional[VoxtralConfig] = None):
        self.config = config or VoxtralConfig()
        self._loaded = False
        self._processor = None
        self._model = None
        self._device = self.config.device
        self._dtype = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        try:
            from transformers import VoxtralForConditionalGeneration, AutoProcessor  # type: ignore
        except Exception as e:  # pragma: no cover - only when voxtral backend is used
            raise ImportError(
                "transformers is required for VoxtralTranscriber. Install transformers >= 4.42."
            ) from e

        import torch  # type: ignore

        model_id = self.config.model_path
        local_only = False  # Allow on-demand downloads

        # Resolve dtype
        self._dtype = _map_dtype(self._device, self.config.dtype)

        # Load processor and model from local files/cache
        self._processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
        self._model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self._dtype,
            device_map=self._device,
            local_files_only=local_only,
        )

        self._loaded = True

    def transcribe(self, audio_pcm: bytes, sample_rate: int, language: Optional[str] = None) -> TranscriptionResult:
        # Lazy-load heavy deps and model
        self._ensure_loaded()

        import torch  # type: ignore

        # Prepare audio tensor
        wav = _pcm16_le_bytes_to_float32(audio_pcm)
        lang = _locale_to_lang(language or self.config.language)

        # Build processor inputs. Voxtral processors accept dict with array + sampling_rate
        inputs = self._processor.apply_transcription_request(
            language=lang or "en",
            audio={"array": wav, "sampling_rate": sample_rate},
            model_id=self.config.model_path,
        )

        # Move to device / dtype
        # Note: On CPU we keep fp32 for stability
        inputs = inputs.to(self._device, dtype=self._dtype if self._device != "cpu" else torch.float32)

        # Generate
        outputs = self._model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)

        # Decode only the newly generated tokens after the prompt length
        prompt_len = inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else 0
        decoded = self._processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)

        text = (decoded[0] if decoded else "").strip()
        duration = len(audio_pcm) / float(2 * max(1, sample_rate)) if audio_pcm else 0.0

        return TranscriptionResult(text=text, language=language or self.config.language or "en-US", duration_sec=duration, confidence=None)
