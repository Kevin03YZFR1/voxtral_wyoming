"""Post-transcription word/phrase replacement."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_LOGGER = logging.getLogger("voxtral_wyoming")


def parse_replacements(
    inline: str | None = None,
    file_path: str | None = None,
) -> dict[str, str]:
    """Parse word replacement definitions from inline string and/or file.

    Returns a dict mapping lowercased source phrases to their replacements.
    Raises ``ValueError`` on malformed entries, ``FileNotFoundError`` if file
    is specified but missing.
    """
    replacements: dict[str, str] = {}

    def _parse_entry(raw: str, source_label: str) -> None:
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            return
        if " -> " not in raw:
            raise ValueError(
                f"Malformed replacement entry ({source_label}): {raw!r}  "
                f"(expected 'source -> replacement')"
            )
        src, repl = raw.split(" -> ", 1)
        src = src.strip()
        repl = repl.strip()
        if not src:
            raise ValueError(
                f"Empty source in replacement entry ({source_label}): {raw!r}"
            )
        replacements[src.lower()] = repl

    if inline:
        for entry in inline.split(","):
            _parse_entry(entry, "WORD_REPLACEMENTS")

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Word replacements file not found: {file_path}"
            )
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            _parse_entry(line, f"{file_path}:{line_no}")

    return replacements


def build_replacement_regex(replacements: dict[str, str]) -> re.Pattern[str] | None:
    """Compile a single regex matching all source phrases.

    Phrases are sorted longest-first so shorter phrases don't shadow longer
    ones.  Returns ``None`` if *replacements* is empty.
    """
    if not replacements:
        return None
    # Sort longest-first to prevent shorter matches shadowing longer ones
    phrases = sorted(replacements.keys(), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b"
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


class TextPostProcessor:
    """Applies word/phrase replacements to transcription output."""

    def __init__(
        self,
        inline: str | None = None,
        file_path: str | None = None,
    ) -> None:
        self._replacements = parse_replacements(inline, file_path)
        self._regex = build_replacement_regex(self._replacements)
        if self._replacements:
            _LOGGER.info(
                "Loaded %d word replacement(s): %s",
                len(self._replacements),
                ", ".join(
                    f"{src!r} -> {repl!r}"
                    for src, repl in self._replacements.items()
                ),
            )
        else:
            _LOGGER.debug("No word replacements configured")

    def apply(self, text: str) -> str:
        """Apply replacements to *text* and return the result."""
        if not self._regex:
            return text
        original = text
        text = self._regex.sub(self._lookup, text)
        # Collapse double-spaces (e.g. from empty replacements / deletions)
        text = re.sub(r"  +", " ", text).strip()
        if text != original:
            _LOGGER.debug("Word replacement: %r -> %r", original, text)
        return text

    def _lookup(self, match: re.Match[str]) -> str:
        replacement = self._replacements[match.group(0).lower()]
        # Preserve leading capital when replacing the first word of the text
        if replacement and match.start() == 0 and match.group(0)[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement
