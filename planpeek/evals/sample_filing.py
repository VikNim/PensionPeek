"""A small, synthetic Form 5500 filing used as a fixture for RAG evaluation.

Real filings need a live download plus Databricks credentials to embed and query --
neither is available in every environment (including the one this was built in). This
fixture is fully offline and deterministic: the "true" page for every fact is known
exactly (because it was authored, not extracted), so `rag_qa_golden.csv` can assert
exact expected pages without depending on a network fetch or PDF-parsing quirks.

Bring your own golden CSV (see `rag_eval.load_golden_dataset`) to evaluate a real
filing instead -- the harness itself doesn't know or care that this one is synthetic.
"""

from __future__ import annotations

from planpeek.models import TextChunk

SAMPLE_FILING_CHUNKS: list[TextChunk] = [
    TextChunk(
        chunk_id="p1-c0",
        page=1,
        text=(
            "Acme Manufacturing 401(k) Plan. Employer Identification Number 12-3456789. "
            "Plan Number 001. Plan Year 2024."
        ),
    ),
    TextChunk(
        chunk_id="p4-c0",
        page=4,
        text=(
            "Schedule H Part I Line 1c(14) Insurance general accounts: beginning of year "
            "$500,000, end of year $520,000. Total assets beginning of year $4,800,000. "
            "Total assets end of year $5,250,000."
        ),
    ),
    TextChunk(
        chunk_id="p5-c0",
        page=5,
        text=(
            "Total income for the plan year was $620,000, consisting primarily of employer "
            "and employee contributions and net investment gain."
        ),
    ),
    TextChunk(
        chunk_id="p5-c1",
        page=5,
        text=(
            "Total expenses for the plan year were $310,000, including benefits paid to "
            "participants and administrative expenses."
        ),
    ),
    TextChunk(
        chunk_id="p9-c0",
        page=9,
        text=(
            "Schedule of Assets (Held at End of Year): Vanguard 500 Index Fund $2,100,000. "
            "Vanguard Total Bond Market Index Fund $900,000. Cash Reserves Federal Money "
            "Market Fund $150,000."
        ),
    ),
]
