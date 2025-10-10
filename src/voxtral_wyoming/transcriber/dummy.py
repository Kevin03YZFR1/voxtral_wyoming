from __future__ import annotations

import logging
from typing import Optional

from .base import ITranscriber, TranscriptionResult
from ..audio import pcm16_duration_seconds

_LOGGER = logging.getLogger(__name__)


class DummyTranscriber:
    """A placeholder transcriber that returns a fixed transcript.

    This lets us wire up the Wyoming server and Home Assistant end-to-end
    before integrating the actual Voxtral model backend.
    """

    def __init__(self, text: str = "transcription not implemented", language: Optional[str] = None):
        self._text = text
        self._language = language

    def transcribe(self, audio_pcm: bytes, sample_rate: int, language: Optional[str] = None) -> TranscriptionResult:
        if not audio_pcm:
            _LOGGER.warning("Empty audio received; returning empty transcript")
            return TranscriptionResult(
                text="",
                language=language or self._language,
                confidence=0.0,
                duration_sec=0.0,
            )

        # Log minimal info; avoid logging audio content
        _LOGGER.debug("Dummy transcribing %d bytes at %d Hz", len(audio_pcm), sample_rate)
        duration = pcm16_duration_seconds(len(audio_pcm), sample_rate, 1)
        return TranscriptionResult(
            text=self._text,
            language=language or self._language,
            confidence=None,
            duration_sec=duration,
        )
