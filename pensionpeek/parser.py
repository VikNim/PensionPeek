from __future__ import annotations

import io
import re
from collections.abc import Iterable

from pensionpeek.context import contains_mask_artifact, sanitize_filing_context
from pensionpeek.models import FilingMetrics, ParsedFiling, TextChunk


class ParsingError(RuntimeError):
    """Raised when a PDF cannot be opened or contains no extractable text."""


MONEY_TOKEN = re.compile(r"(?<![A-Za-z])(?:\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\)?)(?![A-Za-z])")


FALLBACK_ASSET_LABELS: tuple[tuple[str, str], ...] = (
    ("Interest-bearing cash", r"interest[- ]bearing\s+cash"),
    ("Pooled separate accounts", r"pooled\s+separate\s+accounts?"),
    ("Insurance general accounts", r"(?:insurance\s+)?general\s+accounts?"),
    ("Registered investment companies", r"registered\s+investment\s+compan"),
    ("Employer securities", r"employer\s+securities"),
    ("Participant loans", r"participant\s+loans"),
)


def _number(value: str, *, count_field: bool = False) -> float | None:
    raw = value.strip().replace("$", "").replace(",", "").replace(" ", "")
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()")
    if raw == "-123456789012345":
        return None
    if raw.startswith("-123456789012345") and len(raw) > 16:
        # EFAST populated facsimiles prefix many visible monetary fields with
        # a hidden sentinel. The remaining digits are the displayed value.
        raw = raw[16:]
        negative = False
    elif count_field and raw.startswith("12345678") and len(raw) > 8:
        # Participant-count fields use the shorter positive form of the same
        # EFAST sentinel (for example, 1234567811 represents a visible 11).
        raw = raw[8:]
    try:
        parsed = float(raw)
    except ValueError:
        return None
    parsed = -parsed if negative else parsed
    # EFAST facsimiles sometimes contain hidden placeholder identifiers such as
    # -12345678901234567890 behind a visible amount. No Form 5500 money field can
    # plausibly reach this range, so exclude it before matching nearby labels.
    return parsed if abs(parsed) <= 1_000_000_000_000_000 else None


def _values_after_label(
    text: str,
    label_pattern: str,
    line_window: int = 3,
    minimum_values: int = 1,
    count_field: bool = False,
) -> list[float]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    label = re.compile(label_pattern, re.IGNORECASE)
    for index, line in enumerate(lines):
        match = label.search(line)
        if not match:
            continue
        nearby_parts = [line[match.end() :]]
        for next_line in lines[index + 1 : index + line_window]:
            nearby = " ".join(nearby_parts)
            tokens = MONEY_TOKEN.findall(nearby)
            values = [_number(token, count_field=count_field) for token in tokens]
            clean = [value for value in values if value is not None]
            if len(clean) >= minimum_values:
                return clean
            if tokens:
                return clean
            nearby_parts.append(next_line)
        tokens = MONEY_TOKEN.findall(" ".join(nearby_parts))
        values = [_number(token, count_field=count_field) for token in tokens]
        clean = [value for value in values if value is not None]
        if clean:
            return clean
    return []


def _amount_pair(text: str, *labels: str) -> tuple[float | None, float | None]:
    for label in labels:
        values = _values_after_label(text, label, minimum_values=2)
        if len(values) >= 2:
            return values[-2], values[-1]
        if len(values) == 1:
            return None, values[0]
    return None, None


def _single_amount(text: str, *labels: str) -> float | None:
    for label in labels:
        values = _values_after_label(text, label, line_window=1)
        if values:
            return values[-1]
    return None


def _single_count(text: str, *labels: str) -> int | None:
    for label in labels:
        values = _values_after_label(text, label, line_window=1, count_field=True)
        if values:
            amount = values[-1]
            if 0 <= amount <= 100_000_000:
                return int(amount)
    return None


def _schedule_h_text(text: str) -> str:
    match = re.search(r"schedule\s+h", text, re.IGNORECASE)
    return text[match.start() :] if match else text


def _schedule_h_categories(text: str) -> dict[str, float]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]

    def eoy(code: str) -> float:
        escaped = re.escape(code)
        code_pattern = re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
        for line in lines:
            match = code_pattern.search(line)
            if not match:
                continue
            tokens = MONEY_TOKEN.findall(line[match.end() :])
            has_efast_sentinel = any("123456789012345" in token for token in tokens)
            if not has_efast_sentinel:
                # Printed line identifiers also occur in Schedule H instructions.
                # A populated EFAST row carries the hidden numeric sentinel; skip
                # instructional references and let label fallback cover other PDFs.
                continue
            values = [_number(token) for token in tokens]
            clean = [value for value in values if value is not None]
            if len(clean) >= 2:
                return clean[-1]
            if len(clean) == 1:
                return clean[0]
            # A real form row with empty sentinel fields should remain empty,
            # not borrow a value from a later occurrence or neighboring row.
            if tokens:
                return 0.0
        return 0.0

    category_codes: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Cash", ("1a", "1c(1)")),
        ("Receivables", ("1b(1)", "1b(2)", "1b(3)")),
        ("U.S. government securities", ("1c(2)",)),
        ("Corporate debt", ("1c(3)(A)", "1c(3)(B)")),
        ("Corporate stock", ("1c(4)(A)", "1c(4)(B)")),
        ("Other direct investments", ("1c(5)", "1c(6)", "1c(7)")),
        ("Participant loans", ("1c(8)",)),
        ("Common / collective trusts", ("1c(9)",)),
        ("Pooled separate accounts", ("1c(10)",)),
        ("Master trusts and 103-12 entities", ("1c(11)", "1c(12)")),
        ("Registered investment companies", ("1c(13)",)),
        ("Insurance general accounts", ("1c(14)",)),
        ("Other general investments", ("1c(15)",)),
        ("Employer-related investments", ("1d(1)", "1d(2)")),
        ("Property used in plan operations", ("1e",)),
    )
    categories = {name: sum(eoy(code) for code in codes) for name, codes in category_codes}
    categories = {name: value for name, value in categories.items() if value > 0}
    if categories:
        return categories

    # Some third-party PDFs omit printed Schedule H line identifiers. Retain a
    # conservative label-based fallback for a few unambiguous categories.
    fallback: dict[str, float] = {}
    for name, label in FALLBACK_ASSET_LABELS:
        _, value = _amount_pair(text, label)
        if value is not None and value > 0:
            fallback[name] = value
    return fallback


def extract_metrics_from_text(text: str) -> FilingMetrics:
    has_schedule_h = re.search(r"\bschedule\s+h\b", text, re.IGNORECASE) is not None
    schedule_h = _schedule_h_text(text) if has_schedule_h else ""
    assets_boy, assets_eoy = _amount_pair(
        schedule_h,
        r"total\s+assets(?:\s*\(add\s+lines?[^)]*\))?",
        r"assets\s+at\s+beginning\s+and\s+end\s+of\s+year",
    )
    liabilities_boy, liabilities_eoy = _amount_pair(
        schedule_h,
        r"total\s+liabilities(?:\s*\(add\s+lines?[^)]*\))?",
    )
    net_assets_boy, net_assets_eoy = _amount_pair(
        schedule_h,
        r"net\s+assets(?:\s+available\s+for\s+benefits)?",
    )

    categories: dict[str, float] = {}
    if has_schedule_h:
        categories = _schedule_h_categories(schedule_h)

    return FilingMetrics(
        assets_boy=assets_boy,
        assets_eoy=assets_eoy,
        liabilities_boy=liabilities_boy,
        liabilities_eoy=liabilities_eoy,
        net_assets_boy=net_assets_boy,
        net_assets_eoy=net_assets_eoy,
        total_income=(
            _single_amount(
                schedule_h,
                r"total\s+income(?:\s*\(add\s+lines?[^)]*\))?",
            )
            if has_schedule_h
            else None
        ),
        total_expenses=(
            _single_amount(
                schedule_h,
                r"total\s+expenses(?:\s*\(add\s+lines?[^)]*\))?",
            )
            if has_schedule_h
            else None
        ),
        participants_boy=_single_count(
            text,
            r"total\s+number\s+of\s+participants\s+at\s+the\s+beginning\s+of\s+the\s+plan\s+year",
            r"participants\s+(?:at\s+)?beginning\s+of\s+year",
        ),
        participants_eoy=_single_count(
            text,
            r"total\s+number\s+of\s+participants\s+at\s+the\s+end\s+of\s+the\s+plan\s+year",
            r"participants\s+(?:at\s+)?end\s+of\s+year",
        ),
        asset_categories=categories,
    )


def chunk_pages(
    pages: Iterable[str],
    chunk_size: int = 1_000,
    overlap: int = 150,
) -> list[TextChunk]:
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be larger than overlap.")
    chunks: list[TextChunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        text = sanitize_filing_context(page_text)
        if not text:
            continue
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
                if boundary > start + chunk_size // 2:
                    end = boundary + 1
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"p{page_number}-c{chunk_index}",
                        page=page_number,
                        text=chunk_text,
                    )
                )
                chunk_index += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def parse_pdf(pdf_bytes: bytes) -> ParsedFiling:
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ParsingError("The selected file is not a PDF.")
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete installs
        raise ParsingError(
            "PyMuPDF is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        document = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        pages = [page.get_text("text", sort=True).strip() for page in document]
        document.close()
    except Exception as exc:
        raise ParsingError("The PDF is encrypted, damaged, or could not be parsed.") from exc

    if not any(pages):
        raise ParsingError(
            "This filing appears to be image-only and has no extractable text. "
            "OCR is not included in this first version."
        )

    full_text = "\n\n".join(pages)
    warnings: list[str] = []
    if any(contains_mask_artifact(page) for page in pages):
        warnings.append(
            "Hidden EFAST mask and sentinel artifacts were excluded from the AI retrieval context."
        )
    if "schedule h" not in full_text.lower():
        warnings.append(
            "No Schedule H text was detected; this may be a small-plan filing or the schedule "
            "may be image-only."
        )
    metrics = extract_metrics_from_text(full_text)
    if not any(
        value is not None
        for value in (metrics.assets_eoy, metrics.total_income, metrics.total_expenses)
    ):
        warnings.append(
            "Financial totals could not be read confidently from the PDF layout; use the source "
            "pages to verify values."
        )
    return ParsedFiling(
        pages=pages,
        chunks=chunk_pages(pages),
        metrics=metrics,
        warnings=warnings,
    )
