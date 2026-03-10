from __future__ import annotations

"""Voxtral-backed transcriber implementation using Mistral's Voxtral model.

Offline-only execution that loads local model files (no network calls).
"""

import os
import time
from dataclasses import dataclass
from typing import Optional
import logging

from .base import ITranscriber, TranscriptionResult

_logger = logging.getLogger("voxtral_wyoming.transcriber")

# Languages supported by each Voxtral generation
VOXTRAL_GEN1_LANGUAGES = ["en-US", "fr-FR", "de-DE", "es-ES", "it-IT", "pt-PT", "nl-NL", "hi-IN"]
VOXTRAL_GEN2_LANGUAGES = VOXTRAL_GEN1_LANGUAGES + ["ar-SA", "zh-CN", "ja-JP", "ko-KR", "ru-RU"]


def _detect_device() -> str:
    """Automatically detect the best available device for inference.

    Returns:
        Device string: 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, 'cpu' otherwise
    """
    try:
        import torch  # type: ignore

        # Check for NVIDIA GPU
        if torch.cuda.is_available():
            device = "cuda"
            _logger.info(f"Auto-detected device: {device} (NVIDIA GPU available)")
            return device

        # Check for Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            _logger.info(f"Auto-detected device: {device} (Apple Silicon available)")
            return device

        # Fallback to CPU
        device = "cpu"
        _logger.info(f"Auto-detected device: {device} (no GPU available)")
        return device

    except ImportError:
        # PyTorch not available yet, default to CPU
        _logger.warning("PyTorch not available for device detection, defaulting to CPU")
        return "cpu"
    except Exception as e:
        _logger.warning(f"Error during device detection: {e}, defaulting to CPU")
        return "cpu"


@dataclass
class VoxtralConfig:
    model_id: str = None  # type: ignore
    device: str = None  # type: ignore
    dtype: Optional[str] = None
    locale: str = None  # type: ignore
    max_seconds: float = None  # type: ignore
    use_chat_mode: bool = None  # type: ignore
    system_prompt: str = None  # type: ignore
    transcription_delay_ms: Optional[int] = None

    def __post_init__(self):
        """Load values from environment variables if not explicitly provided."""
        if self.model_id is None:
            self.model_id = os.getenv("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
        if self.device is None:
            self.device = os.getenv("DEVICE", "auto")
        if self.dtype is None:
            self.dtype = os.getenv("DATA_TYPE", None)
        if self.locale is None:
            self.locale = os.getenv("LANGUAGE_FALLBACK", "en-US")
        if self.max_seconds is None:
            self.max_seconds = float(os.getenv("MAX_SECONDS", "30"))
        if self.use_chat_mode is None:
            self.use_chat_mode = os.getenv("USE_CHAT_MODE", "false").lower() in ("true", "1", "yes")
        if self.system_prompt is None:
            self.system_prompt = os.getenv(
                "SYSTEM_PROMPT",
                "You are a voice assistant for a smart home. Transcribe the user's voice command accurately. "
                "Commands are typically short, imperative sentences like 'turn on the lights' or 'set temperature to 20 degrees'. "
                "Focus on accuracy and be aware of smart home terminology."
            )
        if self.transcription_delay_ms is None:
            env_val = os.getenv("TRANSCRIPTION_DELAY_MS", None)
            if env_val is not None:
                self.transcription_delay_ms = int(env_val)
        _validate_transcription_delay_ms(self.transcription_delay_ms)


# Valid transcription_delay_ms values: multiples of 80 from 80–1200, plus 2400
_VALID_DELAY_VALUES = set(range(80, 1201, 80)) | {2400}


def _validate_transcription_delay_ms(value: Optional[int]) -> Optional[int]:
    """Validate transcription_delay_ms value.

    Valid values: multiples of 80 from 80 to 1200, or exactly 2400.
    Returns None if input is None (meaning 'use model default').
    """
    if value is None:
        return None
    if value not in _VALID_DELAY_VALUES:
        raise ValueError(
            f"Invalid TRANSCRIPTION_DELAY_MS={value}. "
            f"Must be a multiple of 80 between 80 and 1200, or exactly 2400. "
            f"Recommended value: 480 (best balance of latency and accuracy)."
        )
    return value


def _delay_ms_to_tokens(delay_ms: int) -> int:
    """Convert transcription_delay_ms to num_delay_tokens.

    For Voxtral Realtime models with default audio parameters
    (sampling_rate=16000, hop_length=160, audio_length_per_tok=8),
    each token corresponds to 80ms of audio.
    """
    return delay_ms // 80


def _locale_to_lang(locale: Optional[str]) -> Optional[str]:
    if not locale:
        return None
    # Convert 'en-US' or 'en_US' → 'en'
    return locale.split("-")[0].split("_")[0]


def _map_dtype(dtype_str: Optional[str]):
    """Map dtype string to torch dtype or return None for auto-detection.

    Args:
        dtype_str: Data type string or None for auto-detection

    Returns:
        torch dtype, quantization string, or None for auto-detection
    """
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover - only when voxtral backend is used
        raise ImportError(
            "PyTorch is required for VoxtralTranscriber. Install torch >= 2.3."
        ) from e

    # None or empty means auto-detect from model files
    if dtype_str is None or dtype_str == "" or dtype_str.lower() in ("auto", "none"):
        return None

    norm = dtype_str.lower()

    # Standard torch dtypes
    if norm in ("bf16", "bfloat16"):
        return torch.bfloat16
    if norm in ("fp16", "float16", "f16"):
        return torch.float16
    if norm in ("fp32", "float32"):
        return torch.float32
    if norm in ("fp8", "float8"):
        return torch.float8_e4m3fn  # FP8 E4M3 format (most common)

    _logger.warning(f"Unknown dtype: {dtype_str}. Using auto-detection instead")

    return None


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
        self._is_realtime_model = False

        # Auto-detect device if set to 'auto'
        if self.config.device.lower() == "auto":
            self._device = _detect_device()
        else:
            self._device = self.config.device
            _logger.info(f"Using manually configured device: {self._device}")

        self._dtype = None

        # Preload everything on server startup to prevent slowing down first request
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        try:
            from transformers import AutoConfig, AutoProcessor  # type: ignore
        except Exception as e:  # pragma: no cover - only when voxtral backend is used
            raise ImportError(
                "transformers is required for VoxtralTranscriber. Install transformers >= 5.2."
            ) from e

        import torch  # type: ignore

        model_id = self.config.model_id
        local_only = os.path.isabs(model_id)

        # Resolve dtype - None means auto-detect from model
        self._dtype = _map_dtype(self.config.dtype)

        # Log dtype selection
        if self._dtype is None:
            _logger.info(f"Loading model {model_id} with auto-detected data type from model files")
        else:
            dtype_display = str(self._dtype) if hasattr(self._dtype, '__name__') else self._dtype
            _logger.info(f"Loading model {model_id} with manually specified data type: {dtype_display}")

        # Detect gen1 vs gen2 from model config metadata.
        # Gen2 (realtime) models have model_type="voxtral_realtime",
        # Gen1 (legacy) models have model_type="voxtral".
        model_config = AutoConfig.from_pretrained(model_id, local_files_only=local_only)
        self._is_realtime_model = model_config.model_type == "voxtral_realtime"
        _logger.info(f"Detected {'gen2 realtime' if self._is_realtime_model else 'gen1 legacy'} model (model_type={model_config.model_type!r})")

        # Import the correct model class based on detected generation
        if self._is_realtime_model:
            try:
                from transformers import VoxtralRealtimeForConditionalGeneration  # type: ignore
                _model_cls = VoxtralRealtimeForConditionalGeneration
            except ImportError:
                raise ImportError(
                    f"Model {model_id} is a gen2 realtime model but VoxtralRealtimeForConditionalGeneration "
                    f"is not available. Install transformers >= 5.2."
                )
        else:
            try:
                from transformers import VoxtralForConditionalGeneration  # type: ignore
                _model_cls = VoxtralForConditionalGeneration
            except ImportError:
                raise ImportError(
                    f"Model {model_id} is a gen1 model but VoxtralForConditionalGeneration "
                    f"is not available. Install transformers >= 4.57."
                )

        # Load processor and model
        self._processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)

        model_kwargs = {
            "local_files_only": local_only,
            "device_map": self._device,
        }
        if self._dtype is not None:
            model_kwargs["torch_dtype"] = self._dtype

        try:
            self._model = _model_cls.from_pretrained(model_id, **model_kwargs)
        except Exception as e:
            _logger.warning(f"Failed to load model on {self._device}: {e}. Falling back to CPU")
            self._device = "cpu"
            model_kwargs["device_map"] = "cpu"
            self._model = _model_cls.from_pretrained(model_id, **model_kwargs)

        _logger.info(f"Model loaded using {_model_cls.__name__}")

        # Log the actual dtype that was loaded
        if hasattr(self._model, 'dtype'):
            _logger.info(f"Model loaded successfully with data type: {self._model.dtype}")
        else:
            # Try to infer from first parameter
            try:
                first_param_dtype = next(self._model.parameters()).dtype
                _logger.info(f"Model loaded successfully with data type: {first_param_dtype}")
            except Exception:
                _logger.info(f"Model loaded successfully (data type detection unavailable)")

        # Model is already on the target device via device_map, just set eval mode
        self._model.eval()

        # Apply transcription_delay_ms for Gen2 realtime models.
        # Must be set on the processor's audio config BEFORE processing audio,
        # so that both the audio padding and num_delay_tokens are consistent.
        if self._is_realtime_model and self.config.transcription_delay_ms is not None:
            delay_ms = self.config.transcription_delay_ms
            self._processor.mistral_common_audio_config.transcription_delay_ms = float(delay_ms)
            num_tokens = _delay_ms_to_tokens(delay_ms)
            self._model.config.default_num_delay_tokens = num_tokens
            _logger.info(f"Transcription delay set to {delay_ms}ms ({num_tokens} delay tokens)")
        elif self._is_realtime_model:
            default_tokens = getattr(self._model.config, "default_num_delay_tokens", "unknown")
            _logger.info(f"Using model's default transcription delay ({default_tokens} delay tokens)")
        elif self.config.transcription_delay_ms is not None:
            _logger.warning(
                f"TRANSCRIPTION_DELAY_MS={self.config.transcription_delay_ms} is only supported "
                f"for Gen2 realtime models. Ignoring for Gen1 model."
            )

        if self._is_realtime_model and self.config.use_chat_mode:
            raise ValueError(
                "USE_CHAT_MODE=true is not supported for Gen2 realtime models. "
                "The Gen2 processor does not have a chat template. "
                "Disable chat mode or switch to a Gen1 model."
            )

        self._loaded = True

    @property
    def supported_languages(self) -> list[str]:
        """Return the language list for the loaded model generation."""
        return VOXTRAL_GEN2_LANGUAGES if self._is_realtime_model else VOXTRAL_GEN1_LANGUAGES

    def transcribe(self, audio_pcm: bytes, sample_rate: int, locale: Optional[str] = None) -> TranscriptionResult:
        # Lazy-load heavy deps and model
        self._ensure_loaded()

        import torch  # type: ignore

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

        # Prepare audio as float32 numpy array in [-1, 1]
        wav = _pcm16_le_bytes_to_float32(audio_pcm)

        # Resample if the incoming audio rate doesn't match the processor's expected rate
        target_sr = self._processor.feature_extractor.sampling_rate
        if sample_rate != target_sr:
            import soxr
            _logger.info(f"Resampling audio from {sample_rate}Hz to {target_sr}Hz")
            wav = soxr.resample(wav, sample_rate, target_sr)
            sample_rate = target_sr

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
                if self._is_realtime_model:
                    # Gen2 realtime API: processor takes the audio array directly.
                    # NOTE: Gen2 streaming models do not support language hints.
                    # While TranscriptionRequest has a language field, the streaming
                    # tokenizer in mistral_common ignores it entirely — the token
                    # sequence is identical regardless of the language value.
                    # Language is determined purely by the model from the audio.
                    model_inputs = self._processor(wav, sampling_rate=sample_rate, return_tensors="pt")
                else:
                    # Gen1 legacy API: apply_transcription_request with explicit parameters
                    model_inputs = self._processor.apply_transcription_request(
                        language=language_only,
                        audio=wav,
                        model_id=self.config.model_id,
                        sampling_rate=sample_rate,
                        format=["wav"],  # WAV is the container format for PCM audio data
                    )
            except Exception as e:
                raise RuntimeError(f"Error preparing audio for model: {e}") from e

            # Store input length BEFORE moving to device to ensure we have the correct value
            # apply_transcription_request returns a BatchEncoding object with input_ids
            input_ids_length = model_inputs.get("input_ids").shape[1] if "input_ids" in model_inputs else 0

        # Move inputs to the model's device and dtype.
        # BatchFeature.to() in transformers >= 5.2 handles integer tensors correctly.
        model_inputs = model_inputs.to(self._model.device, dtype=self._model.dtype)

        # Generation settings — greedy decoding per model card (temperature=0).
        # We use do_sample=False instead of passing temperature=0.0, because
        # HuggingFace generate() does not accept temperature as a valid kwarg
        # when sampling is disabled.
        # Gen2 realtime transcribe-only: the model auto-determines output length from audio,
        # so we don't pass max_new_tokens.  Gen1 and chat mode: derive from max_seconds
        # using the 80ms/token formula (e.g. 60s → 750 tokens).
        gen2_transcribe_only = self._is_realtime_model and not self.config.use_chat_mode
        gen_kwargs: dict = {"do_sample": False}
        if not gen2_transcribe_only:
            gen_kwargs["max_new_tokens"] = int(self.config.max_seconds / 0.08)
            gen_kwargs["num_beams"] = 1

        # Time the model inference
        inference_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model.generate(
                **model_inputs,
                **gen_kwargs,
            )
        inference_time = time.perf_counter() - inference_start
        _logger.debug(f"Model inference completed in {inference_time:.2f}s")

        # Decode the outputs.
        # Gen2 realtime transcribe-only: decode full output (official API pattern).
        # Gen1 / chat mode: slice off prompt tokens first.
        try:
            if gen2_transcribe_only:
                decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
            elif input_ids_length > 0:
                decoded = self._processor.batch_decode(
                    outputs[:, input_ids_length:], skip_special_tokens=True
                )
            else:
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
        # Compute duration from original PCM bytes at the original incoming rate.
        # After resampling, sample_rate has been reassigned to the target rate,
        # but audio_pcm still contains bytes at the original rate. Use len(wav)
        # (float32 samples, already resampled) with the current sample_rate instead.
        duration = len(wav) / float(max(1, sample_rate)) if audio_pcm else 0.0

        # Log total transcription time
        total_time = time.perf_counter() - start_time
        _logger.debug(f"Total transcription time: {total_time:.2f}s (audio duration: {duration:.2f}s)")

        return TranscriptionResult(
            text=text,
            language=locale,
            duration_sec=duration,
            confidence=None,
        )
