"""Extraction of Form 5500 Schedule H, Line 4i -- Schedule of Assets (Held at End of Year).

Large plans that hold assets directly (not exclusively through a single pooled trust)
attach this schedule, itemizing every holding by issuer/fund name and year-end value.
It's the only place in a Form 5500 filing where individual fund names appear -- Schedule
H's own asset_categories are grouped by investment *vehicle* type (common/collective
trust, registered investment company, ...), not by asset class, so a plan whose assets
sit mostly in pooled funds is otherwise unclassifiable as equity/bond/cash. When this
schedule is present, real fund names (validated against a live Google LLC 401(k) filing:
"Vanguard 500 Index Fund", "MetWest Total Return Bond Fund", ...) give planpeek.risk
something to actually classify.

Not every filing includes this schedule -- plans invested entirely through a single
master trust file their holdings on the trust's own separate Form 5500 (a "Direct Filing
Entity" filing), which this module does not chase down. When absent, callers get an
empty holdings list, not an error: this is supplementary detail, not required data.
"""

from __future__ import annotations

import re

from planpeek.models import AssetHolding

_SECTION_HEADER = re.compile(
    r"schedule\s+h,?\s+line\s+4i\s*[-–]\s*schedule\s+of\s+assets", re.IGNORECASE
)
_COLUMN_HEADER = re.compile(r"identity\s+of", re.IGNORECASE)
# The DOL's own column-header wording (standardized across every filer, since this is a
# fixed government form) can appear jumbled by PDF column-reading order after "identity
# of" -- e.g. "...(d) Current (a) or Similar Party ... Par or Maturity Value Cost Value".
# Whatever trails the LAST of these header-only phrases is where real holdings start;
# without this, the header noise merges into the first holding's identity text.
_HEADER_NOISE_TAIL = re.compile(
    r"(current\s+value|par\s+or\s+maturity\s+value|cost\s+value)", re.IGNORECASE
)
_HEADER_NOISE_WINDOW = 600
_TRAILING_VALUE = re.compile(r"\**\s*\$?\s*(?P<value>[\d,]{4,}(?:\.\d{2})?)\s*$")
_TOTAL_LINE = re.compile(r"^total\b", re.IGNORECASE)
# Footnote/legend lines that happen to end in something that could look like a number
# (e.g. an EIN) but are never a holding row.
_SKIP_LINE = re.compile(
    r"^(\*+\s*party-in-interest|\*+\s*cost information|see independent auditor)", re.IGNORECASE
)


_MAX_SECTION_CHARS = 20_000


def _schedule_section(full_text: str) -> str | None:
    header_match = _SECTION_HEADER.search(full_text)
    if not header_match:
        return None
    column_match = _COLUMN_HEADER.search(full_text, header_match.end())
    if not column_match:
        return None
    start = column_match.end()
    window = full_text[start : start + _HEADER_NOISE_WINDOW]
    noise_matches = list(_HEADER_NOISE_TAIL.finditer(window))
    if noise_matches:
        start += noise_matches[-1].end()
    # The schedule can appear twice if a filing bundles a duplicated page; only the
    # first occurrence is used. _parse_holdings itself stops at the first "Total"
    # line it hits, so this window only needs to be generous, not exact.
    return full_text[start : start + _MAX_SECTION_CHARS]


def _parse_holdings(section: str) -> tuple[list[AssetHolding], float | None]:
    holdings: list[AssetHolding] = []
    reported_total: float | None = None
    buffer: list[str] = []
    for raw_line in section.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or _SKIP_LINE.match(line):
            continue
        if _TOTAL_LINE.match(line):
            match = _TRAILING_VALUE.search(line)
            if match:
                reported_total = float(match.group("value").replace(",", ""))
            break
        match = _TRAILING_VALUE.search(line)
        if not match:
            buffer.append(line)
            continue
        identity_tail = line[: match.start()].strip(" *$")
        identity = " ".join([*buffer, identity_tail]).strip(" *")
        buffer = []
        if not identity:
            continue
        try:
            value = float(match.group("value").replace(",", ""))
        except ValueError:
            continue
        holdings.append(AssetHolding(identity=identity, value=value))
    return holdings, reported_total


def extract_holdings(full_text: str) -> tuple[list[AssetHolding], list[str]]:
    """Extract itemized holdings from a filing's Schedule of Assets, if present.

    Returns (holdings, warnings). An empty holdings list with no warnings means the
    filing simply doesn't include this schedule -- not every plan is required to file
    one. A non-empty warnings list flags cases worth surfacing to the user, such as the
    extracted holdings not reconciling with the schedule's own printed total (a sign
    some rows -- typically ones with multi-line wrapped descriptions -- were missed or
    mis-split, since this is a heuristic text-column parser, not a structured table).
    """
    section = _schedule_section(full_text)
    if section is None:
        return [], []
    holdings, reported_total = _parse_holdings(section)
    warnings: list[str] = []
    if holdings and reported_total is not None:
        extracted_total = sum(item.value for item in holdings)
        if reported_total > 0:
            drift = abs(extracted_total - reported_total) / reported_total
            if drift > 0.02:
                warnings.append(
                    f"Extracted Schedule of Assets holdings sum to ${extracted_total:,.0f}, "
                    f"which doesn't reconcile with the schedule's own total of "
                    f"${reported_total:,.0f}. Some line items may be missing or "
                    "misread -- verify against the source filing."
                )
    return holdings, warnings
