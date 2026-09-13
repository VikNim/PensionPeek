from __future__ import annotations

import re

_EFAST_NUMERIC_SENTINEL = re.compile(
    r"(?<!\d)-123456789012345(?P<visible_suffix>\d*)(?:\.0+)?(?!\d)"
)
_MASKED_TEXT = re.compile(
    r"\bABCDEFGHI\b|\bA\s+B\s+C\s+D\s+E\s+F\s+G\s+H\s+I\b|(?<![A-Za-z])X{5,}(?![A-Za-z])|\*{5,}"
)


def contains_mask_artifact(text: str) -> bool:
    return bool(_EFAST_NUMERIC_SENTINEL.search(text) or _MASKED_TEXT.search(text))


def sanitize_filing_context(text: str) -> str:
    """Remove EFAST hidden-field artifacts while preserving visible numeric suffixes."""

    def replace_numeric_sentinel(match: re.Match[str]) -> str:
        return match.group("visible_suffix")

    cleaned = _EFAST_NUMERIC_SENTINEL.sub(replace_numeric_sentinel, text)
    cleaned = _MASKED_TEXT.sub("", cleaned)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()
