from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="mcp SDK not installed; run with --extra mcp to test the server")

from pensionpeek import mcp_server as srv  # noqa: E402
from pensionpeek.efast import EfastError  # noqa: E402
from pensionpeek.models import Filing, FilingMetrics, ParsedFiling, RetrievedFiling  # noqa: E402
from pensionpeek.parser import ParsingError  # noqa: E402
from pensionpeek.rag import RagAnswer, RagError  # noqa: E402
from pensionpeek.retrieval import RetrievalError  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_caches():
    srv._filings.clear()
    srv._parsed.clear()
    srv._engines.clear()
    yield
    srv._filings.clear()
    srv._parsed.clear()
    srv._engines.clear()


def _filing(**overrides) -> Filing:
    defaults = dict(
        filing_id="F123",
        plan_name="Acme 401(k)",
        sponsor="Acme LLC",
        ein="123456789",
        plan_number="001",
        plan_year=2024,
        participants_boy=42,
        assets_eoy=1_234_567.0,
    )
    defaults.update(overrides)
    return Filing(**defaults)


def test_digest_reports_no_matches() -> None:
    assert srv._digest([]) == "No matching public filings were found."


def test_digest_includes_filing_id_and_key_fields() -> None:
    digest = srv._digest([_filing()])
    assert "filing_id=F123" in digest
    assert "Acme 401(k)" in digest
    assert "EIN 12-3456789 / Plan 001" in digest
    assert "$1,234,567" in digest
    assert "42" in digest


def test_search_filings_normalizes_search_by_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_by = {}

    def fake_search(value, search_by="Company", limit=100):
        seen_by["search_by"] = search_by
        return [_filing()]

    monkeypatch.setattr(srv._efast, "search", fake_search)
    result = srv.search_filings("Acme", search_by="EIN")

    assert seen_by["search_by"] == "EIN"
    assert "filing_id=F123" in result
    assert srv._filings["F123"].plan_name == "Acme 401(k)"


def test_search_filings_defaults_to_company(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_search(value, search_by="Company", limit=100):
        calls["search_by"] = search_by
        return []

    monkeypatch.setattr(srv._efast, "search", fake_search)
    srv.search_filings("Acme")
    assert calls["search_by"] == "Company"


def test_search_filings_surfaces_efast_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search(value, search_by="Company", limit=100):
        raise EfastError("The Department of Labor search service is not responding right now.")

    monkeypatch.setattr(srv._efast, "search", fake_search)
    assert "not responding" in srv.search_filings("Acme")


def test_get_filing_history_caches_filings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv._efast, "history", lambda ein, plan_number, limit=200: [_filing()])
    result = srv.get_filing_history("123456789", "001")
    assert "filing_id=F123" in result
    assert "F123" in srv._filings


def test_get_filing_financials_reports_unknown_filing_id() -> None:
    assert "Unknown filing_id" in srv.get_filing_financials("does-not-exist")


def test_get_filing_financials_downloads_parses_and_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filing = _filing()
    srv._filings[filing.key] = filing

    retrieved = RetrievedFiling(pdf_bytes=b"%PDF-", filename="f.pdf", source_url="https://x")
    metrics = FilingMetrics(
        assets_boy=100.0,
        assets_eoy=200.0,
        total_income=50.0,
        total_expenses=10.0,
        asset_categories={"Cash": 200.0},
    )
    parsed = ParsedFiling(pages=["p1"], chunks=[], metrics=metrics, warnings=["heads up"])

    monkeypatch.setattr(srv, "download_filing", lambda filing: retrieved)
    monkeypatch.setattr(srv, "parse_pdf", lambda pdf_bytes: parsed)

    result = srv.get_filing_financials(filing.key)

    assert "$100" in result and "$200" in result
    assert "Net income: $40" in result
    assert "Cash: $200" in result
    assert "heads up" in result
    # Second call must not re-download or re-parse.
    monkeypatch.setattr(
        srv,
        "download_filing",
        lambda filing: (_ for _ in ()).throw(AssertionError("should be cached")),
    )
    srv.get_filing_financials(filing.key)


def test_get_filing_financials_surfaces_retrieval_error(monkeypatch: pytest.MonkeyPatch) -> None:
    filing = _filing()
    srv._filings[filing.key] = filing
    monkeypatch.setattr(
        srv,
        "download_filing",
        lambda filing: (_ for _ in ()).throw(RetrievalError("not available")),
    )
    assert srv.get_filing_financials(filing.key) == "not available"


def test_get_filing_financials_surfaces_parsing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    filing = _filing()
    srv._filings[filing.key] = filing
    retrieved = RetrievedFiling(pdf_bytes=b"%PDF-", filename="f.pdf", source_url="https://x")
    monkeypatch.setattr(srv, "download_filing", lambda filing: retrieved)
    monkeypatch.setattr(
        srv,
        "parse_pdf",
        lambda pdf_bytes: (_ for _ in ()).throw(ParsingError("image-only")),
    )
    assert srv.get_filing_financials(filing.key) == "image-only"


def test_ask_filing_reports_unknown_filing_id() -> None:
    assert "Unknown filing_id" in srv.ask_filing("nope", "What are the assets?")


def test_ask_filing_builds_engine_once_and_returns_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filing = _filing()
    retrieved = RetrievedFiling(pdf_bytes=b"%PDF-", filename="f.pdf", source_url="https://x")
    parsed = ParsedFiling(pages=["p1"], chunks=[], metrics=FilingMetrics())
    srv._filings[filing.key] = filing
    srv._parsed[filing.key] = (retrieved, parsed)

    build_calls = []

    class FakeEngine:
        def ask(self, question, conversation=None):
            return RagAnswer(
                text=f"Answer to: {question}",
                citations=[{"page": 3, "chunk_id": "p3-c0", "excerpt": "supporting text"}],
            )

    def fake_engine_ctor(**kwargs):
        build_calls.append(kwargs)
        return FakeEngine()

    monkeypatch.setattr(srv, "RagEngine", fake_engine_ctor)

    result = srv.ask_filing(filing.key, "What are total assets?")

    assert "Answer to: What are total assets?" in result
    assert "p.3 (p3-c0): supporting text" in result
    assert len(build_calls) == 1
    assert filing.key in srv._engines

    # A second question against the same filing must reuse the cached engine.
    srv.ask_filing(filing.key, "And liabilities?")
    assert len(build_calls) == 1


def test_ask_filing_surfaces_rag_error_from_engine_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filing = _filing()
    retrieved = RetrievedFiling(pdf_bytes=b"%PDF-", filename="f.pdf", source_url="https://x")
    parsed = ParsedFiling(pages=["p1"], chunks=[], metrics=FilingMetrics())
    srv._filings[filing.key] = filing
    srv._parsed[filing.key] = (retrieved, parsed)

    def fake_engine_ctor(**kwargs):
        raise RagError("Enter an API key for Databricks.")

    monkeypatch.setattr(srv, "RagEngine", fake_engine_ctor)
    result = srv.ask_filing(filing.key, "What are total assets?")
    assert result == "Enter an API key for Databricks."


def test_evict_oldest_closes_and_bounds_cache() -> None:
    closed = []

    class Closeable:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    cache: dict[str, Closeable] = {}
    for i in range(srv.MAX_CACHED_FILINGS + 3):
        cache[str(i)] = Closeable(str(i))
        srv._evict_oldest(cache)

    assert len(cache) == srv.MAX_CACHED_FILINGS
    assert closed == ["0", "1", "2"]


def test_glossary_index_lists_all_terms() -> None:
    index = srv.glossary_index()
    assert "Schedule H:" in index
    assert "EFAST2:" in index


def test_glossary_term_lookup_is_case_insensitive() -> None:
    assert srv.glossary_term("schedule h").startswith("The financial schedule")


def test_glossary_term_unknown_returns_message() -> None:
    assert "No glossary entry found" in srv.glossary_term("not-a-real-term")
