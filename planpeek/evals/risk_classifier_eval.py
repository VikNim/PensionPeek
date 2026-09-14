"""Precision/recall/F1 evaluation for planpeek.risk's name-based classifier.

Same pattern as the Week 4 Session 1 routing-agent evaluation: a golden dataset of
(input, expected label) pairs, run through the classifier, scored with a full
classification_report rather than raw accuracy. Accuracy alone would be misleading
here for the same reason it's misleading for an imbalanced router: a classifier that
always guesses "unclassified" would still score well on some real 401(k) plans, since
"unclassified" genuinely is the correct answer for a large share of Schedule H vehicle
types (see risk.py's own module docstring).

Two rows in the golden dataset (tagged "known_miss" in the CSV's source column) are
documented, intentional classifier misses -- Wellesley Income Fund (a balanced fund
whose name pattern-matches "income fund", the same pattern that correctly identifies
pure bond funds) and Fidelity VIP Contrafund (a well-known growth-equity fund whose
name contains no asset-class keyword at all). These are kept in the golden dataset
deliberately, not excluded: a report showing < 100% is more honest than one rigged to
look perfect, and it's the same limitation risk.py already documents.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from planpeek.evals.metrics import ClassificationReport, classification_report
from planpeek.risk import classify_label

GOLDEN_DATASET_PATH = Path(__file__).parent / "data" / "risk_classifier_golden.csv"


@dataclass(frozen=True, slots=True)
class GoldenRow:
    label_text: str
    expected_asset_class: str
    source: str = ""


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[GoldenRow]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            GoldenRow(
                label_text=row["label_text"],
                expected_asset_class=row["expected_asset_class"],
                source=row.get("source", ""),
            )
            for row in csv.DictReader(f)
        ]
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def evaluate(golden: list[GoldenRow] | None = None) -> ClassificationReport:
    golden = golden if golden is not None else load_golden_dataset()
    y_true = [row.expected_asset_class for row in golden]
    y_pred = [classify_label(row.label_text) for row in golden]
    return classification_report(y_true, y_pred)


def main() -> None:
    golden = load_golden_dataset()
    report = evaluate(golden)
    print("=" * 60)
    print("  planpeek.risk.classify_label() -- golden dataset evaluation")
    print("=" * 60)
    print(report.summary())
    print()
    print(f"Rows: {len(golden)}")
    known_misses = [row for row in golden if row.source.startswith("known_miss")]
    if known_misses:
        print(f"Includes {len(known_misses)} intentional, documented misclassification(s):")
        for row in known_misses:
            predicted = classify_label(row.label_text)
            print(
                f"  - {row.label_text!r}: expected {row.expected_asset_class!r}, "
                f"got {predicted!r}"
            )


if __name__ == "__main__":
    main()
