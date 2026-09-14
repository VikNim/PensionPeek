"""Evaluation harness for planpeek.rag.RagEngine answers.

Two metrics, mirroring the Week 4 Session 2 agentic-RAG tutorial:

- **Citation accuracy** (deterministic, no LLM): RagEngine's system prompt requires
  citing pages in the exact form ``[p. N]``. This checks whether the pages an answer
  actually cites match the golden dataset's expected pages -- the same shape as that
  tutorial's ``citation_f1`` (expected [Tag]s vs. tags found in the answer text), but
  built against page numbers instead of tool-derived tags, since that's what
  RagEngine's own citation contract is.
- **Retrieval accuracy** (deterministic, no LLM): the same expected-vs-actual
  comparison, but against the pages the Chroma query actually retrieved
  (``RagAnswer.citations``) rather than what the model chose to cite in prose -- these
  can differ (retrieval can find the right page while the model fails to cite it, or
  vice versa), which is exactly why the tutorial scores context_precision/recall
  separately from citation accuracy.
- **Faithfulness** (LLM-as-judge, optional): delegates to ``rag_judge.judge_faithfulness``.
  Not run unless a judge callable is supplied, since it requires a live model.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from planpeek.evals.metrics import set_precision_recall_f1
from planpeek.evals.rag_judge import FaithfulnessVerdict

if TYPE_CHECKING:
    from planpeek.rag import RagAnswer, RagEngine

GOLDEN_DATASET_PATH = Path(__file__).parent / "data" / "rag_qa_golden.csv"
CITED_PAGE_PATTERN = re.compile(r"\[p\.\s*(\d+)\]")

JudgeFn = Callable[[str, list[str]], FaithfulnessVerdict]


@dataclass(frozen=True, slots=True)
class RagEvalCase:
    question: str
    expected_pages: set[int]


@dataclass(frozen=True, slots=True)
class RagEvalResult:
    case: RagEvalCase
    answer_text: str
    cited_pages: set[int]
    retrieved_pages: set[int]
    citation_precision: float
    citation_recall: float
    citation_f1: float
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    faithfulness: FaithfulnessVerdict | None = None


def parse_cited_pages(answer_text: str) -> set[int]:
    """Pages cited in an answer's text via RagEngine's required `[p. N]` format."""
    return {int(match) for match in CITED_PAGE_PATTERN.findall(answer_text)}


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[RagEvalCase]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            RagEvalCase(
                question=row["question"],
                expected_pages={int(p) for p in row["expected_pages"].split("|") if p.strip()},
            )
            for row in csv.DictReader(f)
        ]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def evaluate_answer(
    case: RagEvalCase,
    answer: RagAnswer,
    *,
    judge: JudgeFn | None = None,
) -> RagEvalResult:
    cited_pages = parse_cited_pages(answer.text)
    retrieved_pages = {citation["page"] for citation in answer.citations}

    c_p, c_r, c_f1 = set_precision_recall_f1(case.expected_pages, cited_pages)
    r_p, r_r, r_f1 = set_precision_recall_f1(case.expected_pages, retrieved_pages)

    faithfulness = None
    if judge is not None:
        excerpts = [citation["excerpt"] for citation in answer.citations]
        faithfulness = judge(answer.text, excerpts)

    return RagEvalResult(
        case=case,
        answer_text=answer.text,
        cited_pages=cited_pages,
        retrieved_pages=retrieved_pages,
        citation_precision=c_p,
        citation_recall=c_r,
        citation_f1=c_f1,
        retrieval_precision=r_p,
        retrieval_recall=r_r,
        retrieval_f1=r_f1,
        faithfulness=faithfulness,
    )


def run_eval(
    engine: RagEngine,
    golden: list[RagEvalCase],
    *,
    judge: JudgeFn | None = None,
) -> list[RagEvalResult]:
    """Run every golden question through `engine.ask()` and score each answer."""
    results = []
    for case in golden:
        answer = engine.ask(case.question)
        results.append(evaluate_answer(case, answer, judge=judge))
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def scorecard(results: list[RagEvalResult]) -> str:
    groups: dict[str, list[float]] = {
        "citation_precision": [r.citation_precision for r in results],
        "citation_recall": [r.citation_recall for r in results],
        "citation_f1": [r.citation_f1 for r in results],
        "retrieval_precision": [r.retrieval_precision for r in results],
        "retrieval_recall": [r.retrieval_recall for r in results],
        "retrieval_f1": [r.retrieval_f1 for r in results],
    }
    lines = ["=" * 48, "  RagEngine -- evaluation scorecard", "=" * 48]
    for name, values in groups.items():
        v = _mean(values)
        lines.append(f"  {name:<20} {v:.2f}  {'#' * int(v * 10)}")

    judged = [r for r in results if r.faithfulness is not None]
    if judged:
        supported = sum(1 for r in judged if r.faithfulness.label == "SUPPORTED")
        lines.append(f"  {'faithfulness':<20} {supported}/{len(judged)} SUPPORTED")
    lines.append("=" * 48)
    return "\n".join(lines)


def per_row_table(results: list[RagEvalResult]) -> str:
    header = f"{'question':<55} {'cite_f1':>8} {'retr_f1':>8} {'faithful':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        faithful = r.faithfulness.label if r.faithfulness else "-"
        question = r.case.question[:53] + (".." if len(r.case.question) > 53 else "")
        lines.append(f"{question:<55} {r.citation_f1:>8.3f} {r.retrieval_f1:>8.3f} {faithful:>10}")
    return "\n".join(lines)


def main() -> None:
    """Run the bundled golden dataset against the bundled sample filing.

    Requires the same Databricks environment variables as planpeek.rag (see README).
    Uses the synthetic fixture in sample_filing.py, so results reflect the harness and
    the configured model's real behavior -- not a live DOL filing.
    """
    import os

    from planpeek.evals.rag_judge import judge_faithfulness
    from planpeek.evals.sample_filing import SAMPLE_FILING_CHUNKS
    from planpeek.rag import RagEngine, resolve_databricks_settings

    settings = resolve_databricks_settings(os.getenv)
    engine = RagEngine(
        filing_key="planpeek-evals-sample-filing",
        chunks=SAMPLE_FILING_CHUNKS,
        provider="Databricks",
        api_key=settings["api_key"],
        model_name=settings["model_name"],
        embedding_backend="Databricks",
        embedding_model=settings["embedding_model"],
        databricks_base_url=settings["base_url"],
        databricks_profile=settings["profile"],
    )
    try:
        golden = load_golden_dataset()
        results = run_eval(engine, golden, judge=judge_faithfulness)
    finally:
        engine.close()

    print(per_row_table(results))
    print()
    print(scorecard(results))


if __name__ == "__main__":
    main()
