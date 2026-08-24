from __future__ import annotations

import re
from typing import Any

import requests

from pensionpeek.models import Filing

SEARCH_URL = "https://www.efast.dol.gov/services/afs"
DEFAULT_TIMEOUT = (5, 25)
USER_AGENT = "PensionPeek/0.1 (+public Form 5500 research tool)"


class EfastError(RuntimeError):
    """Raised when the public EFAST search service cannot be used."""


def normalize_ein(value: str) -> str:
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) != 9:
        raise ValueError("An EIN must contain exactly 9 digits.")
    return digits


def _escape_lucene_phrase(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip())
    if not cleaned:
        raise ValueError("Enter a company name or EIN to search.")
    if len(cleaned) > 160:
        raise ValueError("Search text must be 160 characters or fewer.")
    return cleaned.replace("\\", "\\\\").replace('"', '\\"')


def build_query(value: str, search_by: str) -> str:
    if search_by == "EIN":
        return f"ein:{normalize_ein(value)}"
    phrase = _escape_lucene_phrase(value)
    return f'(planname:"{phrase}") OR (plansponsor:"{phrase}")'


class EfastClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.get(SEARCH_URL, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise EfastError(
                "The Department of Labor search service is not responding right now. "
                "Try again in a moment."
            ) from exc
        except ValueError as exc:
            raise EfastError(
                "The Department of Labor returned an unreadable search response."
            ) from exc

        if not isinstance(payload, dict) or "hits" not in payload:
            raise EfastError("The Department of Labor returned an unexpected search response.")
        return payload

    def search(self, value: str, search_by: str = "Company", limit: int = 100) -> list[Filing]:
        query = build_query(value, search_by)
        payload = self._request(
            {
                "q.parser": "lucene",
                "size": min(max(limit, 1), 200),
                "sort": "planyear desc",
                "q": query,
            }
        )
        hits = payload.get("hits", {}).get("hit", []) or []
        return [Filing.from_hit(hit) for hit in hits]

    def history(self, ein: str, plan_number: str, limit: int = 200) -> list[Filing]:
        normalized_ein = normalize_ein(ein)
        pn = re.sub(r"\D", "", plan_number)
        query = f"ein:{normalized_ein}"
        if pn:
            query += f" AND pn:{pn.zfill(3)}"
        payload = self._request(
            {
                "q.parser": "lucene",
                "size": min(max(limit, 1), 200),
                "sort": "planyear asc",
                "q": query,
            }
        )
        hits = payload.get("hits", {}).get("hit", []) or []
        filings = [Filing.from_hit(hit) for hit in hits]
        return sorted(filings, key=lambda item: item.plan_year or 0)
