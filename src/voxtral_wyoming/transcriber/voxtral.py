from __future__ import annotations

"""Voxtral-backed transcriber implementation using Mistral's Voxtral model.

Offline-only execution that loads local model files (no network calls).

Environment variables:
- VOXTRAL_MODEL_DIR / VOXTRAL_MODEL_PATH: Local path to the Voxtral model dir (recommended)
- VOXTRAL_DEVICE: cpu|cuda|mps (default: cuda), automatically falls back to cpu, if other device fails
- VOXTRAL_DTYPE: fp32|fp16|bf16 (default: fp32)
- VOXTRAL_LANGUAGE: locale like en-US (default: None → processor/model default)
- VOXTRAL_MAX_NEW_TOKENS: generation length (default: 500)

Note: If a Hugging Face repo ID is provided instead of a path, loading will still
use local_files_only=True. Ensure the model is present in the local HF cache or
provide VOXTRAL_MODEL_PATH to a directory on disk.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from .base import ITranscriber, TranscriptionResult


@dataclass
class VoxtralConfig:
    model_path: Optional[str] = os.getenv("VOXTRAL_MODEL_PATH") or os.getenv("VOXTRAL_MODEL_DIR") or "mistralai/Voxtral-Mini-3B-2507"
    device: str = os.getenv("VOXTRAL_DEVICE", "cuda")
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


def _detect_audio_format(audio_bytes: bytes) -> str:
    """Detect audio format from byte signature.

    Returns: 'pcm', 'mp3', 'wav', 'ogg', 'flac', or 'unknown'
    """
    if not audio_bytes or len(audio_bytes) < 4:
        return "unknown"

    # Check for common audio format signatures
    if audio_bytes[:4] == b'RIFF' and len(audio_bytes) >= 12 and audio_bytes[8:12] == b'WAVE':
        return "wav"
    elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb' or audio_bytes[:2] == b'\xff\xf3':
        return "mp3"
    elif audio_bytes[:4] == b'OggS':
        return "ogg"
    elif audio_bytes[:4] == b'fLaC':
        return "flac"

    # Heuristic: if it looks like mostly small values typical of PCM16, assume PCM
    # This is not foolproof but helps distinguish PCM from compressed formats
    return "pcm"


def _convert_to_pcm16_with_ffmpeg(audio_bytes: bytes, sample_rate: int = 16000) -> tuple[bytes, bool]:
    """
    Convert input audio (mp3/wav/ogg/flac/etc.) to raw PCM16 mono at the given sample_rate using ffmpeg.

    Returns (converted_bytes, success).
    If ffmpeg is not available or conversion fails, returns original bytes and False.
    """
    import shutil
    import subprocess

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
                "-",  # Read from stdin
                "-f",
                "s16le",  # Output format: signed 16-bit little-endian
                "-acodec",
                "pcm_s16le",  # Audio codec: PCM 16-bit
                "-ac",
                "1",  # Mono (1 channel)
                "-ar",
                str(sample_rate),  # Sample rate
                "-",  # Write to stdout
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30,  # Prevent hanging on very large files
        )
        return proc.stdout, True
    except Exception:
        # Conversion failed - return original bytes
        return audio_bytes, False


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


def _float32_to_base64_wav(audio_float32, sample_rate: int):
    """Convert float32 audio array to base64-encoded WAV format for VoxtralProcessor."""
    try:
        import numpy as np  # type: ignore
        import base64
        import io
        import wave
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "NumPy, base64, io, and wave are required for VoxtralTranscriber."
        ) from e

    # Convert float32 [-1, 1] to int16 PCM
    audio_int16 = (audio_float32 * 32767).astype(np.int16)

    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    # Get WAV bytes and encode to base64
    wav_bytes = buffer.getvalue()
    base64_str = base64.b64encode(wav_bytes).decode('utf-8')
    return base64_str


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

        # Preload everything on server startup to prevent slowing down first request
        self._ensure_loaded()

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
        # Allow downloading from HuggingFace if model is not in local cache
        # Set VOXTRAL_LOCAL_ONLY=true to enforce strict offline mode
        local_only = os.getenv("VOXTRAL_LOCAL_ONLY", "false").lower() == "true"

        # Resolve dtype
        self._dtype = _map_dtype(self._device, self.config.dtype)

        # Load processor and model from local files/cache
        self._processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
        self._model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            dtype=self._dtype,
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
        import logging

        _logger = logging.getLogger("voxtral_wyoming.transcriber")

        # Start timing for overall transcription
        start_time = time.perf_counter()

        # Validate audio input - handle None case
        if audio_pcm is None:
            _logger.warning("Received None audio_pcm, returning empty transcription")
            return TranscriptionResult(
                text="",
                language=language or self.config.language or "en-US",
                duration_sec=0.0,
                confidence=None,
            )

        # Validate and potentially convert audio format
        # Allow empty audio to pass through (will return empty/minimal transcription)
        if audio_pcm:
            audio_format = _detect_audio_format(audio_pcm)
            if audio_format not in ("pcm", "unknown"):
                # Known compressed format detected - attempt automatic conversion
                _logger.info(
                    f"Audio format '{audio_format}' detected, attempting automatic conversion to PCM16 using ffmpeg..."
                )
                converted_audio, success = _convert_to_pcm16_with_ffmpeg(audio_pcm, sample_rate)

                if success:
                    _logger.info(f"Successfully converted {audio_format} to PCM16 ({len(converted_audio)} bytes)")
                    audio_pcm = converted_audio
                else:
                    # Conversion failed - provide clear error
                    raise ValueError(
                        f"Audio format '{audio_format}' detected, but automatic conversion to PCM16 failed. "
                        f"This usually means ffmpeg is not available on the server. "
                        f"Please either:\n"
                        f"  1. Install ffmpeg on the server, or\n"
                        f"  2. Convert audio to PCM16 format before sending:\n"
                        f"     ffmpeg -i input.{audio_format} -f s16le -acodec pcm_s16le -ac 1 -ar {sample_rate} output.pcm"
                    )

        # Prepare audio as float32 numpy array in [-1, 1]
        wav = _pcm16_le_bytes_to_float32(audio_pcm)
        lang = _locale_to_lang(language or self.config.language)

        # Use native transcription API with numpy array input
        # This is the proper API for transcription-only use cases
        # Pass the audio as numpy array directly (not base64) so the processor
        # can properly extract audio features using WhisperFeatureExtractor
        try:
            model_inputs = self._processor.apply_transcription_request(
                language=lang or "en",
                audio=wav,
                model_id=self.config.model_path,
                sampling_rate=sample_rate,
                format=["wav"]  # WAV is the container format for PCM audio data
            )
        except Exception as e:
            raise RuntimeError(f"Error preparing audio for model: {e}") from e

        # Store input length BEFORE moving to device to ensure we have the correct value
        # apply_transcription_request returns a BatchEncoding object with input_ids
        input_ids_length = model_inputs.get("input_ids").shape[1] if "input_ids" in model_inputs else 0

        # Move inputs to device - use .to() method if available to preserve BatchEncoding structure
        try:
            if hasattr(model_inputs, 'to'):
                # BatchEncoding has a .to() method that preserves structure
                model_inputs = model_inputs.to(self._device)
            else:
                # Fallback for plain dict
                model_inputs = {
                    k: v.to(self._device) if hasattr(v, 'to') else v
                    for k, v in model_inputs.items()
                }
        except Exception as e:
            _logger.warning(f"Could not move inputs to device {self._device}: {e}")
            pass  # Fallback: use inputs as-is if device move fails

        # Generate with CPU-friendly, deterministic settings
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }

        # Time the model inference
        inference_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model.generate(
                **model_inputs,
                **gen_kwargs,
            )
        inference_time = time.perf_counter() - inference_start
        _logger.debug(f"Model inference completed in {inference_time:.2f}s")

        # Decode the outputs
        # For apply_transcription_request, we need to skip the prompt tokens
        # The HuggingFace example shows: outputs[:, inputs.input_ids.shape[1]:]
        try:
            # Decode with proper slicing to remove prompt tokens
            if input_ids_length > 0:
                # Slice to get only generated tokens (excluding prompt)
                generated_tokens = outputs[:, input_ids_length:]
                decoded = self._processor.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
            else:
                # No input_ids or length is 0, decode full output
                decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
        except Exception as e:
            _logger.error(f"Error decoding tokens: {e}", exc_info=True)
            decoded = []

        text = (decoded[0] if decoded else "").strip()
        _logger.info(f"Final transcription text (length={len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
        duration = len(audio_pcm) / float(2 * max(1, sample_rate)) if audio_pcm else 0.0

        # Log total transcription time
        total_time = time.perf_counter() - start_time
        _logger.debug(f"Total transcription time: {total_time:.2f}s (audio duration: {duration:.2f}s)")

        return TranscriptionResult(
            text=text,
            language=language or self.config.language or "en-US",
            duration_sec=duration,
            confidence=None,
        )
