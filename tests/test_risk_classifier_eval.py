from __future__ import annotations

from planpeek.evals.risk_classifier_eval import (
    GOLDEN_DATASET_PATH,
    GoldenRow,
    evaluate,
    load_golden_dataset,
)
from planpeek.risk import classify_label


def test_golden_dataset_file_loads_and_is_nonempty() -> None:
    golden = load_golden_dataset()
    assert GOLDEN_DATASET_PATH.exists()
    assert len(golden) > 10
    assert all(isinstance(row, GoldenRow) for row in golden)


def test_golden_dataset_expected_labels_are_all_valid_asset_classes() -> None:
    valid = {"equity", "fixed_income", "cash", "mixed", "other", "unclassified"}
    golden = load_golden_dataset()
    for row in golden:
        assert row.expected_asset_class in valid, row.label_text


def test_evaluate_scores_a_perfect_synthetic_dataset_as_perfect() -> None:
    golden = [
        GoldenRow("S&P 500 Index Fund", "equity"),
        GoldenRow("Total Bond Market Fund", "fixed_income"),
        GoldenRow("Cash Reserves Money Market Fund", "cash"),
    ]
    report = evaluate(golden)
    assert report.accuracy == 1.0


def test_evaluate_on_bundled_golden_dataset_reflects_documented_known_misses() -> None:
    golden = load_golden_dataset()
    report = evaluate(golden)

    # The bundled dataset intentionally includes two documented classifier misses
    # (see risk_classifier_eval.py's module docstring and risk.py's own docstring) --
    # accuracy should be high but not perfect, and should not silently regress to
    # "everything is unclassified" or similar degenerate cases.
    assert 0.85 <= report.accuracy < 1.0
    assert report.macro_f1 > 0.8

    known_miss_rows = [
        row
        for row in golden
        if row.source.startswith("known_miss") and row.label_text == "Fidelity VIP Contrafund"
    ]
    assert len(known_miss_rows) == 1
    assert classify_label(known_miss_rows[0].label_text) != known_miss_rows[0].expected_asset_class
