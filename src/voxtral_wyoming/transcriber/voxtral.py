from __future__ import annotations

from pycountry import languages

"""Voxtral-backed transcriber implementation using Mistral's Voxtral model.

Offline-only execution that loads local model files (no network calls).
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from .base import ITranscriber, TranscriptionResult


@dataclass
class VoxtralConfig:
    model_id: str = os.getenv("MODEL_ID", "mistralai/Voxtral-Mini-3B-2507")
    device: str = os.getenv("DEVICE", "cuda")
    dtype: str = os.getenv("DATA_TYPE", "bf16")  # fp32|fp16|bf16 - see .env.example for trade-offs
    locale: str = os.getenv("LANGUAGE_FALLBACK", "en-US")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "128"))
    use_chat_mode: bool = os.getenv("USE_CHAT_MODE", "false").lower() in ("true", "1", "yes")
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a voice assistant for a smart home. Transcribe the user's voice command accurately. "
        "Commands are typically short, imperative sentences like 'turn on the lights' or 'set temperature to 20 degrees'. "
        "Focus on accuracy and be aware of smart home terminology."
    )


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
    if norm in ("fp32", "float32"):
        return torch.float32

    import logging
    _logger = logging.getLogger("voxtral_wyoming.transcriber")
    _logger.error(f"Unknown dtype: {dtype_str}. Falling back to fp32")

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


def _audio_array_to_base64(audio_array, sample_rate: int) -> str:
    """Convert numpy audio array to base64-encoded WAV format for chat template."""
    import base64
    import io
    try:
        import soundfile as sf  # type: ignore
    except Exception as e:
        raise ImportError(
            "soundfile is required for chat mode. Install soundfile."
        ) from e

    # Write audio to in-memory WAV file
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)

    # Encode to base64
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode('utf-8')


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

        model_id = self.config.model_id
        local_only = False

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

    def transcribe(self, audio_pcm: bytes, sample_rate: int, locale: Optional[str] = None) -> TranscriptionResult:
        # Lazy-load heavy deps and model
        self._ensure_loaded()

        import torch  # type: ignore
        import logging

        _logger = logging.getLogger("voxtral_wyoming.transcriber")

        # Start timing for overall transcription
        start_time = time.perf_counter()

        locale = locale or self.config.locale
        language_only = _locale_to_lang(locale)

        # Validate audio input - handle None case
        if audio_pcm is None:
            _logger.warning("Received None audio_pcm, returning empty transcription")
            return TranscriptionResult(
                text="",
                language=locale,
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

        # Log transcription start with key parameters
        mode_str = "chat mode" if self.config.use_chat_mode else "transcribe-only mode"
        _logger.info(
            f"Starting transcription ({mode_str}): language={locale}, sample_rate={sample_rate}Hz, model_id={self.config.model_id}"
        )

        # Choose between chat mode (with system prompt) or transcribe-only mode
        if self.config.use_chat_mode:
            # Chat mode: Use apply_chat_template with system prompt
            # This allows custom prompts to guide the transcription context
            _logger.debug(f"Using chat mode with system prompt: {self.config.system_prompt[:100]}...")

            try:
                # Convert audio to base64 for chat template
                audio_base64 = _audio_array_to_base64(wav, sample_rate)

                # Build conversation with system context and transcription request
                # Note: Mistral models may not support explicit "system" role,
                # so we include the system prompt in the user message
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.system_prompt},
                            {"type": "audio", "base64": audio_base64},
                            {"type": "text", "text": "Transcribe the audio."},
                        ],
                    },
                ]

                # Apply chat template with language hint if available
                # Note: MistralCommonTokenizer does not support add_generation_prompt parameter
                model_inputs = self._processor.apply_chat_template(
                    conversation,
                    return_dict=True,
                    tokenize=True,
                )
            except Exception as e:
                raise RuntimeError(f"Error preparing audio for model in chat mode: {e}") from e

            # Store input length for chat mode
            input_ids_length = model_inputs.get("input_ids").shape[1] if "input_ids" in model_inputs else 0
        else:
            # Transcribe-only mode: Use native transcription API (default, current behavior)
            # This is the proper API for transcription-only use cases
            # Pass the audio as numpy array directly (not base64) so the processor
            # can properly extract audio features using WhisperFeatureExtractor
            try:
                model_inputs = self._processor.apply_transcription_request(
                    language=language_only,
                    audio=wav,
                    model_id=self.config.model_id,
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

        # In chat mode, the model sometimes quotes the transcribed text
        # Remove leading and trailing quotes
        if self.config.use_chat_mode and text:
            # Strip quotes iteratively to handle multiple layers
            while text and text[0] in ('"', "'") and text[-1] in ('"', "'") and text[0] == text[-1]:
                text = text[1:-1].strip()

        _logger.info(f"Final transcription text (length={len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
        duration = len(audio_pcm) / float(2 * max(1, sample_rate)) if audio_pcm else 0.0

        # Log total transcription time
        total_time = time.perf_counter() - start_time
        _logger.debug(f"Total transcription time: {total_time:.2f}s (audio duration: {duration:.2f}s)")

        return TranscriptionResult(
            text=text,
            language=locale,
            duration_sec=duration,
            confidence=None,
        )
