from __future__ import annotations

import pytest

from planpeek.evals.metrics import classification_report, set_precision_recall_f1


def test_classification_report_perfect_predictions() -> None:
    report = classification_report(["a", "b", "a"], ["a", "b", "a"])
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    for m in report.per_class:
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0


def test_classification_report_computes_precision_recall_per_label() -> None:
    # 3 true "a", 1 predicted "a" correctly, 1 "a" predicted as "b", 1 "b" predicted as "a".
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "b", "a", "a"]
    report = classification_report(y_true, y_pred)
    by_label = {m.label: m for m in report.per_class}

    # label "a": TP=2, FP=1 (the true "b" predicted "a"), FN=1 (true "a" predicted "b")
    assert by_label["a"].precision == pytest.approx(2 / 3)
    assert by_label["a"].recall == pytest.approx(2 / 3)
    assert by_label["a"].support == 3

    # label "b": TP=0, FP=1, FN=1
    assert by_label["b"].precision == 0.0
    assert by_label["b"].recall == 0.0
    assert by_label["b"].support == 1


def test_classification_report_rejects_empty_or_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="empty"):
        classification_report([], [])
    with pytest.raises(ValueError, match="same length"):
        classification_report(["a"], ["a", "b"])


def test_classification_report_summary_is_printable_text() -> None:
    report = classification_report(["a", "b"], ["a", "a"])
    text = report.summary()
    assert "accuracy" in text
    assert "macro avg f1" in text


def test_set_precision_recall_f1_both_empty_is_vacuously_correct() -> None:
    assert set_precision_recall_f1(set(), set()) == (1.0, 1.0, 1.0)


def test_set_precision_recall_f1_one_empty_is_a_full_miss() -> None:
    assert set_precision_recall_f1({1, 2}, set()) == (0.0, 0.0, 0.0)
    assert set_precision_recall_f1(set(), {1}) == (0.0, 0.0, 0.0)


def test_set_precision_recall_f1_partial_overlap() -> None:
    precision, recall, f1 = set_precision_recall_f1({1, 2, 3}, {2, 3, 4})
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_set_precision_recall_f1_no_overlap_at_all() -> None:
    assert set_precision_recall_f1({1, 2}, {3, 4}) == (0.0, 0.0, 0.0)
