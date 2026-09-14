"""Shared, dependency-free evaluation metrics.

Hand-rolled instead of adding scikit-learn/pandas: everything needed here (multi-class
precision/recall/F1, a confusion matrix, set-based precision/recall/F1 for citation
checking) is a few lines of arithmetic. This matches how the rest of the codebase
avoids a dependency where the stdlib does the job (e.g. vul_parser.py's HTML table
walker uses stdlib html.parser rather than BeautifulSoup).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    per_class: list[ClassMetrics]
    accuracy: float
    macro_f1: float
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)

    def summary(self) -> str:
        header = f"{'label':<24} {'precision':>9} {'recall':>9} {'f1':>9} {'support':>8}"
        lines = [header, "-" * len(header)]
        for m in self.per_class:
            lines.append(
                f"{m.label:<24} {m.precision:>9.3f} {m.recall:>9.3f} {m.f1:>9.3f} {m.support:>8}"
            )
        total_support = sum(m.support for m in self.per_class)
        lines.append("-" * len(header))
        lines.append(f"{'accuracy':<24} {'':>9} {'':>9} {self.accuracy:>9.3f} {total_support:>8}")
        lines.append(f"{'macro avg f1':<24} {'':>9} {'':>9} {self.macro_f1:>9.3f}")
        return "\n".join(lines)


def classification_report(y_true: list[str], y_pred: list[str]) -> ClassificationReport:
    """Per-class precision/recall/F1 plus accuracy and macro-F1 -- no sklearn required."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        raise ValueError("Cannot score an empty dataset")

    labels = sorted(set(y_true) | set(y_pred))
    confusion = Counter(zip(y_true, y_pred, strict=True))

    per_class: list[ClassMetrics] = []
    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(count for (t, p), count in confusion.items() if p == label and t != label)
        fn = sum(count for (t, p), count in confusion.items() if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(count for (t, _p), count in confusion.items() if t == label)
        per_class.append(
            ClassMetrics(label=label, precision=precision, recall=recall, f1=f1, support=support)
        )

    accuracy = sum(count for (t, p), count in confusion.items() if t == p) / len(y_true)
    macro_f1 = sum(m.f1 for m in per_class) / len(per_class) if per_class else 0.0

    return ClassificationReport(
        per_class=per_class, accuracy=accuracy, macro_f1=macro_f1, confusion=dict(confusion)
    )


def set_precision_recall_f1(expected: set, actual: set) -> tuple[float, float, float]:
    """Precision/recall/F1 between two sets (e.g. expected vs. cited page numbers).

    Mirrors the shape of the Week 4 Session 2 citation_f1 metric: both empty is
    vacuously correct (nothing expected, nothing produced -- not a failure); expected
    XOR actual empty is a full miss; otherwise standard set-overlap precision/recall.
    """
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    if not expected or not actual:
        return 0.0, 0.0, 0.0
    tp = len(expected & actual)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / len(actual)
    recall = tp / len(expected)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
