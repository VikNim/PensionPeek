from __future__ import annotations

import pytest
import requests

from planpeek.edgar import EdgarClient, EdgarError, _filing_from_search_hit, edgar_user_agent

SEARCH_HIT = {
    "_id": "0000726865-26-000669:initialn6.htm",
    "_source": {
        "ciks": ["0001048607"],
        "display_names": [
            "LINCOLN LIFE FLEXIBLE PREMIUM VARIABLE LIFE ACCOUNT M  (CIK 0001048607)"
        ],
        "adsh": "0000726865-26-000669",
        "form": "N-6",
        "file_date": "2026-08-06",
    },
}

SUBMISSIONS_PAYLOAD = {
    "name": "LINCOLN LIFE FLEXIBLE PREMIUM VARIABLE LIFE ACCOUNT M",
    "filings": {
        "recent": {
            "form": ["N-6", "N-6/A", "10-K"],
            "accessionNumber": [
                "0000726865-26-000669",
                "0000726865-25-000010",
                "0000726865-25-999999",
            ],
            "filingDate": ["2026-08-06", "2025-01-01", "2025-06-01"],
            "primaryDocument": ["initialn6.htm", "n6a.htm", "10k.htm"],
        }
    },
}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.url = "https://example.com"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.headers: dict = {}
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        return _FakeResponse(self.payload)


class _RaisingSession:
    headers: dict = {}

    def get(self, *args, **kwargs):
        raise requests.ConnectionError("boom")


def test_edgar_user_agent_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    assert "set EDGAR_USER_AGENT" in edgar_user_agent()


def test_edgar_user_agent_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_USER_AGENT", "Acme Research admin@acme.com")
    assert edgar_user_agent() == "Acme Research admin@acme.com"


def test_filing_from_search_hit_parses_real_shape() -> None:
    filing = _filing_from_search_hit(SEARCH_HIT)
    assert filing is not None
    assert filing.cik == "1048607"
    assert filing.accession_number == "0000726865-26-000669"
    assert filing.document_filename == "initialn6.htm"
    assert filing.form_type == "N-6"
    assert filing.file_date == "2026-08-06"
    assert filing.accession_no_dashes == "0000726865-26-000669".replace("-", "")
    assert filing.key == filing.accession_number


def test_filing_from_search_hit_returns_none_when_incomplete() -> None:
    assert _filing_from_search_hit({"_id": "", "_source": {"ciks": []}}) is None


def test_search_parses_hits_into_vul_filings() -> None:
    payload = {"hits": {"hits": [SEARCH_HIT]}}
    session = _FakeSession(payload)
    client = EdgarClient(session=session)

    results = client.search("Lincoln Life")

    assert len(results) == 1
    assert results[0].display_name.startswith("LINCOLN LIFE")
    call = session.calls[0]
    assert call["params"]["forms"] == "N-6"
    assert '"Lincoln Life"' == call["params"]["q"]


def test_search_rejects_empty_query() -> None:
    client = EdgarClient(session=_FakeSession({"hits": {"hits": []}}))
    with pytest.raises(ValueError):
        client.search("   ")


def test_search_wraps_connection_errors() -> None:
    client = EdgarClient(session=_RaisingSession())
    with pytest.raises(EdgarError, match="not responding"):
        client.search("Lincoln Life")


def test_history_filters_to_requested_form_and_sorts_newest_first() -> None:
    session = _FakeSession(SUBMISSIONS_PAYLOAD)
    client = EdgarClient(session=session)

    results = client.history("1048607")

    assert [r.form_type for r in results] == ["N-6", "N-6/A"]
    assert results[0].file_date == "2026-08-06"
    assert results[1].file_date == "2025-01-01"
    assert all(r.display_name == SUBMISSIONS_PAYLOAD["name"] for r in results)


def test_history_rejects_non_numeric_cik() -> None:
    client = EdgarClient(session=_FakeSession(SUBMISSIONS_PAYLOAD))
    with pytest.raises(ValueError):
        client.history("not-a-cik")
