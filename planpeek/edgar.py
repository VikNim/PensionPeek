"""SEC EDGAR search client for Variable Universal Life (VUL) separate-account filings.

Mirrors pensionpeek.efast's shape (search / history over one HTTP client) for a second,
unrelated public filing source: SEC EDGAR's free full-text search API and per-company
submissions index, scoped to Form N-6 / N-6/A -- the filing type used by insurance
company separate accounts that offer variable life insurance policies.

SEC requires every request to sec.gov / data.sec.gov / efts.sec.gov to declare a User-Agent
that identifies the requester (see https://www.sec.gov/os/webmaster-faq#developers).
Generic or missing values risk being rate-limited or blocked. Set EDGAR_USER_AGENT to a
real "Your Company Name admin@yourcompany.com" string before using this outside local
experimentation -- the bundled default is a placeholder, not a real contact.
"""

from __future__ import annotations

import os
import re
from typing import Any

import requests

from pensionpeek.models import VulFiling

FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
DEFAULT_TIMEOUT = (5, 25)
DEFAULT_USER_AGENT = "PensionPeek research-tool (set EDGAR_USER_AGENT to a real contact)"
VUL_FORM_TYPE = "N-6"


class EdgarError(RuntimeError):
    """Raised when SEC EDGAR's public search or submissions service cannot be used."""


def edgar_user_agent() -> str:
    return os.getenv("EDGAR_USER_AGENT", "").strip() or DEFAULT_USER_AGENT


def _clean_query(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip())
    if not cleaned:
        raise ValueError("Enter a carrier or separate account name to search.")
    if len(cleaned) > 160:
        raise ValueError("Search text must be 160 characters or fewer.")
    return cleaned


def _filing_from_search_hit(hit: dict[str, Any]) -> VulFiling | None:
    source = hit.get("_source", {})
    ciks = source.get("ciks") or []
    display_names = source.get("display_names") or []
    accession = source.get("adsh", "")
    doc_id = str(hit.get("_id", ""))
    _accession_prefix, _sep, filename = doc_id.partition(":")
    if not (ciks and accession and filename):
        return None
    return VulFiling(
        cik=str(ciks[0]).lstrip("0") or "0",
        accession_number=accession,
        form_type=str(source.get("form", VUL_FORM_TYPE)),
        file_date=str(source.get("file_date", "")),
        display_name=str(display_names[0]) if display_names else "Unknown separate account",
        document_filename=filename,
    )


class EdgarClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": edgar_user_agent(), "Accept": "application/json"}
        )

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise EdgarError(
                "The SEC EDGAR service is not responding right now. Try again in a moment."
            ) from exc
        except ValueError as exc:
            raise EdgarError("SEC EDGAR returned an unreadable response.") from exc

    def search(
        self, query: str, forms: str = VUL_FORM_TYPE, limit: int = 40
    ) -> list[VulFiling]:
        """Full-text search across EDGAR filings, scoped to the given form type."""
        cleaned = _clean_query(query)
        payload = self._get(
            FULL_TEXT_SEARCH_URL,
            {"q": f'"{cleaned}"', "forms": forms},
        )
        hits = payload.get("hits", {}).get("hits", []) or []
        filings = [_filing_from_search_hit(hit) for hit in hits]
        return [filing for filing in filings if filing is not None][: max(limit, 1)]

    def history(self, cik: str, forms: str = VUL_FORM_TYPE) -> list[VulFiling]:
        """All filings of the given form type for one company, newest first."""
        digits = "".join(char for char in cik if char.isdigit())
        if not digits:
            raise ValueError("Enter a numeric SEC CIK.")
        padded = digits.zfill(10)
        payload = self._get(SUBMISSIONS_URL.format(cik=padded), {})
        recent = payload.get("filings", {}).get("recent", {})
        display_name = str(payload.get("name", "Unknown filer"))
        fields = zip(
            recent.get("form", []),
            recent.get("accessionNumber", []),
            recent.get("filingDate", []),
            recent.get("primaryDocument", []),
            strict=False,
        )
        filings = [
            VulFiling(
                cik=digits.lstrip("0") or "0",
                accession_number=accession,
                form_type=form,
                file_date=file_date,
                display_name=display_name,
                document_filename=primary_document,
            )
            for form, accession, file_date, primary_document in fields
            if form.startswith(forms)
        ]
        return sorted(filings, key=lambda item: item.file_date, reverse=True)
