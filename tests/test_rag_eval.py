from __future__ import annotations

from planpeek.evals.rag_eval import (
    GOLDEN_DATASET_PATH,
    RagEvalCase,
    evaluate_answer,
    load_golden_dataset,
    parse_cited_pages,
    per_row_table,
    run_eval,
    scorecard,
)
from planpeek.evals.rag_judge import FaithfulnessVerdict
from planpeek.rag import RagAnswer


def test_parse_cited_pages_extracts_all_page_citations() -> None:
    text = "Assets were $5.25M [p. 4]. Income was $620K [p. 5], expenses [p. 5]."
    assert parse_cited_pages(text) == {4, 5}


def test_parse_cited_pages_returns_empty_set_when_no_citations() -> None:
    assert parse_cited_pages("No citations in this answer at all.") == set()


def test_golden_dataset_file_loads_and_is_nonempty() -> None:
    golden = load_golden_dataset()
    assert GOLDEN_DATASET_PATH.exists()
    assert len(golden) >= 3
    assert all(isinstance(case.expected_pages, set) for case in golden)
    assert all(case.expected_pages for case in golden)


def test_evaluate_answer_perfect_citation_and_retrieval() -> None:
    case = RagEvalCase(question="What are total assets?", expected_pages={4})
    answer = RagAnswer(
        text="Total assets were $5.25M [p. 4].",
        citations=[{"page": 4, "chunk_id": "p4-c0", "excerpt": "Total assets end of year..."}],
    )
    result = evaluate_answer(case, answer)
    assert result.cited_pages == {4}
    assert result.retrieved_pages == {4}
    assert result.citation_f1 == 1.0
    assert result.retrieval_f1 == 1.0
    assert result.faithfulness is None


def test_evaluate_answer_flags_a_hallucinated_citation() -> None:
    # Model cites page 7, but nothing on page 7 was ever retrieved -- and page 4
    # (the real answer) wasn't cited at all.
    case = RagEvalCase(question="What are total assets?", expected_pages={4})
    answer = RagAnswer(
        text="Total assets were $5.25M [p. 7].",
        citations=[{"page": 4, "chunk_id": "p4-c0", "excerpt": "Total assets end of year..."}],
    )
    result = evaluate_answer(case, answer)
    assert result.cited_pages == {7}
    assert result.citation_f1 == 0.0  # cited the wrong page
    assert result.retrieval_f1 == 1.0  # retrieval itself found the right page


def test_evaluate_answer_runs_judge_when_provided() -> None:
    case = RagEvalCase(question="What are total assets?", expected_pages={4})
    answer = RagAnswer(
        text="Total assets were $5.25M [p. 4].",
        citations=[{"page": 4, "chunk_id": "p4-c0", "excerpt": "Total assets end of year..."}],
    )

    def fake_judge(answer_text: str, excerpts: list[str]) -> FaithfulnessVerdict:
        assert answer_text == answer.text
        assert excerpts == ["Total assets end of year..."]
        return FaithfulnessVerdict(label="SUPPORTED", reasoning="Matches the excerpt.")

    result = evaluate_answer(case, answer, judge=fake_judge)
    assert result.faithfulness is not None
    assert result.faithfulness.label == "SUPPORTED"


class _FakeEngine:
    def __init__(self, answers: dict[str, RagAnswer]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def ask(self, question: str) -> RagAnswer:
        self.asked.append(question)
        return self.answers[question]


def test_run_eval_calls_engine_once_per_golden_case() -> None:
    golden = [
        RagEvalCase(question="Q1", expected_pages={1}),
        RagEvalCase(question="Q2", expected_pages={2}),
    ]
    engine = _FakeEngine(
        {
            "Q1": RagAnswer(
                text="A1 [p. 1].", citations=[{"page": 1, "chunk_id": "c1", "excerpt": ""}]
            ),
            "Q2": RagAnswer(
                text="A2 [p. 2].", citations=[{"page": 2, "chunk_id": "c2", "excerpt": ""}]
            ),
        }
    )
    results = run_eval(engine, golden)
    assert engine.asked == ["Q1", "Q2"]
    assert [r.citation_f1 for r in results] == [1.0, 1.0]


def test_scorecard_and_per_row_table_do_not_crash_on_results() -> None:
    case = RagEvalCase(question="What are total assets?", expected_pages={4})
    answer = RagAnswer(
        text="Total assets were $5.25M [p. 4].",
        citations=[{"page": 4, "chunk_id": "p4-c0", "excerpt": "x"}],
    )
    results = [evaluate_answer(case, answer)]
    assert "evaluation scorecard" in scorecard(results)
    assert "What are total assets?" in per_row_table(results)


def test_scorecard_handles_empty_results() -> None:
    assert "evaluation scorecard" in scorecard([])
