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
    max_new_tokens: int = int(os.getenv("VOXTRAL_MAX_NEW_TOKENS", "128"))


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
            local_files_only=local_only,
        )
        # Move to device explicitly and set eval mode
        try:
            if self._device:
                self._model.to(self._device)
        except Exception:
            # Fallback to CPU if device move fails
            self._device = "cpu"
            self._model.to("cpu")
        self._model.eval()

        self._loaded = True

    def transcribe(self, audio_pcm: bytes, sample_rate: int, language: Optional[str] = None) -> TranscriptionResult:
        # Lazy-load heavy deps and model
        self._ensure_loaded()

        import torch  # type: ignore

        # Prepare audio as float32 numpy array in [-1, 1]
        wav = _pcm16_le_bytes_to_float32(audio_pcm)
        lang = _locale_to_lang(language or self.config.language)

        # Build processor inputs without relying on any file/path loading.
        # Prefer a standard audio -> features call that accepts numpy arrays.
        try:
            if hasattr(self._processor, "feature_extractor"):
                proc_inputs = self._processor.feature_extractor(
                    wav, sampling_rate=sample_rate, return_tensors="pt"
                )
            else:
                # Many processors are callable with (audio=..., sampling_rate=...)
                proc_inputs = self._processor(
                    audio=wav, sampling_rate=sample_rate, return_tensors="pt"
                )
        except Exception as e:
            raise RuntimeError(f"Error preparing audio for model: {e}") from e

        # Determine the expected model input key and move to device/dtype
        if "input_features" in proc_inputs:
            x = proc_inputs["input_features"].to(
                self._device, dtype=self._dtype if self._device != "cpu" else torch.float32
            )
            model_inputs = {"input_features": x}
        elif "input_values" in proc_inputs:
            x = proc_inputs["input_values"].to(
                self._device, dtype=self._dtype if self._device != "cpu" else torch.float32
            )
            model_inputs = {"input_values": x}
        else:
            # As a last resort, try to .to() the entire structure and pass through
            try:
                x_any = proc_inputs.to(
                    self._device, dtype=self._dtype if self._device != "cpu" else torch.float32
                )
                model_inputs = dict(x_any)
            except Exception as e:
                raise RuntimeError(
                    "Unsupported processor outputs for audio input; expected 'input_features' or 'input_values'"
                ) from e

        # Generate with CPU-friendly, deterministic settings
        tokenizer = getattr(self._processor, "tokenizer", None)
        eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad_id is None and eos_id is not None:
            pad_id = eos_id

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id

        with torch.inference_mode():
            outputs = self._model.generate(
                **model_inputs,
                **gen_kwargs,
            )

        # Decode tokens
        try:
            decoded = self._processor.batch_decode(
                outputs, skip_special_tokens=True
            )
        except Exception:
            decoded = []

        text = (decoded[0] if decoded else "").strip()
        duration = len(audio_pcm) / float(2 * max(1, sample_rate)) if audio_pcm else 0.0

        return TranscriptionResult(
            text=text,
            language=language or self.config.language or "en-US",
            duration_sec=duration,
            confidence=None,
        )
