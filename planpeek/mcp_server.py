"""PlanPeek MCP server.

Exposes Form 5500 search, retrieval, and grounded Q&A as MCP tools so any MCP client
(Claude Desktop, Claude Code, etc.) can look up and ask questions about public DOL
filings -- the same EFAST2 data and RAG pipeline the Streamlit app uses, without the UI.

Tools are intent-shaped rather than a 1:1 mirror of the internal modules: a caller
never handles a PDF or a vector store, only a short filing_id (returned by
search_filings / get_filing_history) and plain-language questions.

This process caches parsed filings and RAG engines in memory for its lifetime (one
server process per client session, launched over stdio), mirroring how the Streamlit
app scopes state to a browser session. A bounded cache keeps memory use predictable.

Run:  uv run python -m planpeek.mcp_server
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from planpeek.efast import EfastClient, EfastError
from planpeek.glossary import TERMS
from planpeek.models import Filing, ParsedFiling, RetrievedFiling
from planpeek.parser import ParsingError, parse_pdf
from planpeek.rag import RagEngine, RagError, resolve_databricks_settings
from planpeek.retrieval import RetrievalError, download_filing

MAX_CACHED_FILINGS = 10

mcp = FastMCP("PlanPeek")

_efast = EfastClient()

# filing_id -> Filing, populated by search/history so later tools can reference a
# result without re-searching.
_filings: dict[str, Filing] = {}
# filing_id -> (RetrievedFiling, ParsedFiling), populated on first financial lookup
# or question so a given filing is downloaded and parsed at most once.
_parsed: dict[str, tuple[RetrievedFiling, ParsedFiling]] = {}
# filing_id -> RagEngine, one ephemeral Chroma collection per filing.
_engines: dict[str, RagEngine] = {}


def _remember_filings(filings: list[Filing]) -> None:
    for filing in filings:
        _filings[filing.key] = filing


def _evict_oldest(cache: dict) -> None:
    # Plain dicts preserve insertion order; re-inserting an existing key doesn't
    # change its position, so this is a simple FIFO cap, not true LRU -- enough to
    # keep a long-running server's memory (parsed PDFs, Chroma collections) bounded.
    while len(cache) > MAX_CACHED_FILINGS:
        oldest_id = next(iter(cache))
        evicted = cache.pop(oldest_id)
        if hasattr(evicted, "close"):
            evicted.close()


def _digest(filings: list[Filing]) -> str:
    if not filings:
        return "No matching public filings were found."
    lines = [f"{len(filings)} matching filing(s):"]
    for filing in filings:
        assets = f"${filing.assets_eoy:,.0f}" if filing.assets_eoy is not None else "not reported"
        participants = (
            f"{filing.participants_boy:,}"
            if filing.participants_boy is not None
            else "not reported"
        )
        lines.append(
            f"- filing_id={filing.key} | {filing.plan_name} ({filing.sponsor}) | "
            f"EIN {filing.formatted_ein} / Plan {filing.plan_number} | "
            f"plan year {filing.plan_year or 'n/a'} | assets EOY {assets} | "
            f"participants BOY {participants}"
        )
    return "\n".join(lines)


@mcp.tool()
def search_filings(query: str, search_by: str = "company") -> str:
    """Search public DOL EFAST2 Form 5500 filings by company/plan sponsor name or EIN.

    Args:
        query: A company name, plan sponsor name, or 9-digit EIN.
        search_by: Either "company" or "ein". Defaults to "company".

    Returns:
        A digest of matching filings, each tagged with a filing_id to pass to
        get_filing_financials or ask_filing.
    """
    normalized_by = "EIN" if search_by.strip().lower() == "ein" else "Company"
    try:
        filings = _efast.search(query, search_by=normalized_by)
    except (EfastError, ValueError) as exc:
        return str(exc)
    _remember_filings(filings)
    return _digest(filings)


@mcp.tool()
def get_filing_history(ein: str, plan_number: str) -> str:
    """Get the plan-year filing history for one EIN and three-digit plan number.

    Args:
        ein: The plan sponsor's 9-digit EIN (formatting such as dashes is ignored).
        plan_number: The plan's 3-digit number distinguishing it from the sponsor's
            other plans.

    Returns:
        A digest of filings across years, each tagged with a filing_id.
    """
    try:
        filings = _efast.history(ein, plan_number)
    except (EfastError, ValueError) as exc:
        return str(exc)
    _remember_filings(filings)
    return _digest(filings)


def _get_filing(filing_id: str) -> Filing | str:
    filing = _filings.get(filing_id)
    if filing is None:
        return (
            f"Unknown filing_id {filing_id!r}. Call search_filings or get_filing_history "
            "first to find one."
        )
    return filing


def _ensure_parsed(filing: Filing) -> tuple[RetrievedFiling, ParsedFiling] | str:
    cached = _parsed.get(filing.key)
    if cached:
        return cached
    try:
        retrieved = download_filing(filing)
        parsed = parse_pdf(retrieved.pdf_bytes)
    except (RetrievalError, ParsingError) as exc:
        return str(exc)
    _parsed[filing.key] = (retrieved, parsed)
    _evict_oldest(_parsed)
    return retrieved, parsed


def _money(value: float | None) -> str:
    return f"${value:,.0f}" if value is not None else "not reported"


def _count(value: int | None) -> str:
    return f"{value:,}" if value is not None else "not reported"


@mcp.tool()
def get_filing_financials(filing_id: str) -> str:
    """Download and parse a filing's PDF, returning its Schedule H financial detail.

    Args:
        filing_id: A filing_id returned by search_filings or get_filing_history.

    Returns:
        Extracted assets, liabilities, net assets, income/expenses, participant
        counts, and reported asset categories, plus any parsing warnings. Values are
        heuristically extracted from the PDF and should be verified against the
        source filing before relying on them.
    """
    filing = _get_filing(filing_id)
    if isinstance(filing, str):
        return filing
    result = _ensure_parsed(filing)
    if isinstance(result, str):
        return result
    _retrieved, parsed = result
    metrics = parsed.metrics

    lines = [
        f"{filing.plan_name} -- plan year {filing.plan_year or 'not reported'}",
        f"Assets: {_money(metrics.assets_boy)} (BOY) -> {_money(metrics.assets_eoy)} (EOY)",
        f"Liabilities: {_money(metrics.liabilities_boy)} (BOY) -> "
        f"{_money(metrics.liabilities_eoy)} (EOY)",
        f"Net assets: {_money(metrics.net_assets_boy)} (BOY) -> "
        f"{_money(metrics.net_assets_eoy)} (EOY)",
        f"Total income: {_money(metrics.total_income)}",
        f"Total expenses: {_money(metrics.total_expenses)}",
        f"Net income: {_money(metrics.net_income)}",
        f"Participants: {_count(metrics.participants_boy)} (BOY) -> "
        f"{_count(metrics.participants_eoy)} (EOY)",
    ]
    if metrics.asset_categories:
        lines.append("Reported asset categories:")
        for name, value in sorted(metrics.asset_categories.items(), key=lambda item: -item[1]):
            lines.append(f"  - {name}: {_money(value)}")
    if parsed.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in parsed.warnings)
    lines.append(
        "These are aggregate plan-level figures heuristically extracted from the PDF -- "
        "verify against the source filing before relying on them."
    )
    return "\n".join(lines)


@mcp.tool()
def ask_filing(filing_id: str, question: str) -> str:
    """Ask a grounded question about one filing's content, with page citations.

    Downloads and parses the filing PDF on first use, then answers only from
    retrieved filing excerpts using the configured Databricks model. Requires the
    same Databricks environment variables as the Streamlit app (see README):
    DATABRICKS_FM_BASE_URL, DATABRICKS_PROFILE, LLM_MODEL, EMBEDDING_MODEL, and
    either an OAuth profile session or DATABRICKS_FM_TOKEN.

    Args:
        filing_id: A filing_id returned by search_filings or get_filing_history.
        question: A question about the filing's content.

    Returns:
        A grounded answer citing filing pages, followed by the supporting excerpts.
    """
    filing = _get_filing(filing_id)
    if isinstance(filing, str):
        return filing
    result = _ensure_parsed(filing)
    if isinstance(result, str):
        return result
    _retrieved, parsed = result

    engine = _engines.get(filing_id)
    if engine is None:
        settings = resolve_databricks_settings(os.getenv)
        try:
            engine = RagEngine(
                filing_key=filing.key,
                chunks=parsed.chunks,
                provider="Databricks",
                api_key=settings["api_key"],
                model_name=settings["model_name"],
                embedding_backend="Databricks",
                embedding_model=settings["embedding_model"],
                databricks_base_url=settings["base_url"],
                databricks_profile=settings["profile"],
            )
        except RagError as exc:
            return str(exc)
        _engines[filing_id] = engine
        _evict_oldest(_engines)

    try:
        answer = engine.ask(question)
    except RagError as exc:
        return str(exc)

    lines = [answer.text, "", "Sources:"]
    for citation in answer.citations:
        lines.append(f"- p.{citation['page']} ({citation['chunk_id']}): {citation['excerpt']}")
    return "\n".join(lines)


@mcp.resource("config://glossary")
def glossary_index() -> str:
    """Static resource: every plain-language term PlanPeek defines."""
    return "\n".join(f"{term}: {definition}" for term, definition in TERMS.items())


@mcp.resource("docs://glossary/{term}")
def glossary_term(term: str) -> str:
    """Templated resource: look up one glossary term by name (e.g. 'Schedule H')."""
    for key, definition in TERMS.items():
        if key.lower() == term.lower():
            return definition
    return f"No glossary entry found for {term!r}."


if __name__ == "__main__":
    mcp.run()
