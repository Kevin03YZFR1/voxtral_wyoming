from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    confidence: Optional[float] = None


class ITranscriber(Protocol):
    """Interface for transcribers used by the Wyoming server backend.

    Implementations should be local-only and must not use any remote APIs.
    """

    def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe raw PCM16 mono audio.

        Args:
            audio_pcm: Little-endian PCM16 mono bytes.
            sample_rate: Sample rate in Hz (e.g., 16000).
            language: Optional language/locale hint (e.g., "en-US").
            prompt: Optional context/prompt to guide transcription (e.g., expected vocabulary, topic, formatting hints).

        Returns:
            TranscriptionResult with text and optional metadata.
        """
        ...
