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

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported language locale codes (e.g. ["en-US", "fr-FR"])."""
        ...

    def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int,
        locale: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe raw PCM16 mono audio.

        Args:
            audio_pcm: Little-endian PCM16 mono bytes.
            sample_rate: Sample rate in Hz (e.g., 16000).
            locale: Optional language/locale hint (e.g., "en-US").

        Returns:
            TranscriptionResult with text and optional metadata.
        """
        ...
