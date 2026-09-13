"""Equity / fixed-income / cash classification and a risk-exposure summary.

Shared by both plan types: Form 5500 Schedule of Assets holdings (401k, via
planpeek.schedule_of_assets) and VUL sub-funds / allocation-chart entries (life
insurance, via planpeek.vul_parser). Classification is name-based -- there is no
ticker/CUSIP lookup anywhere in this pipeline -- so it only ever asserts what a
holding's own name plausibly states, and reports whatever it can't confidently place
as unclassified rather than guessing.

This is a heuristic, not a substitute for reading the actual fund prospectus. Two
known failure modes, found by testing against real fund names (a live Google LLC
401(k) filing): a name containing "Income Fund" is classified fixed_income, which is
right for a pure bond fund (PIMCO Income Fund) but wrong for a balanced fund that
happens to use the same naming convention (Vanguard Wellesley Income Fund, actually
~35-65 stock/bond) -- there is no way to tell those apart from the name alone. Rule
order matters: more specific patterns (bond, cash, target-date) are checked before the
broad equity catch-all, so e.g. "Total International Bond Index Fund" resolves to
fixed_income despite also matching equity-ish words like "international" and "index".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

AssetClass = Literal["equity", "fixed_income", "cash", "mixed", "other", "unclassified"]

# Order matters: checked top to bottom, first match wins.
_CLASSIFICATION_RULES: tuple[tuple[AssetClass, re.Pattern[str]], ...] = (
    (
        "cash",
        re.compile(
            r"\b(money\s*market|cash\s*reserves|stable\s*value|short[- ]term\s*reserve)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fixed_income",
        re.compile(
            r"\b(bond|fixed\s*income|treasury|government\s*securities?|total\s*return|"
            r"income\s*fund|debt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "mixed",
        re.compile(
            r"\b(target\s*(date|retirement)|balanced|income\s*and\s*growth|lifecycle|"
            r"asset\s*allocation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "equity",
        re.compile(
            r"\b(index|large\s*cap|small\s*cap|mid\s*cap|smid\s*cap|growth\s*fund|"
            r"value\s*fund|equity|stock|international|emerging\s*markets|real\s*estate)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "other",
        re.compile(
            r"\b(loans?|notes?\s*receivable|self[- ]directed|brokerage|participant\s*loans?)\b",
            re.IGNORECASE,
        ),
    ),
)

RISK_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Conservative", 0.0, 30.0),
    ("Moderate", 30.0, 65.0),
    ("Growth", 65.0, 85.0),
    ("Aggressive", 85.0, 100.001),
)

# A risk band is only shown when at least this fraction of total value was classified
# into equity/fixed_income/cash/mixed -- otherwise a confident-looking label would be
# built on mostly-guessed data, which is worse than no label at all.
MIN_CLASSIFIED_SHARE_FOR_BAND = 0.6
# Target-date and balanced funds count toward the equity side of a risk band at this
# weight, since they hold a mix rather than being purely growth or purely defensive.
MIXED_EQUITY_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class ClassifiedAmount:
    label: str
    asset_class: AssetClass
    value: float


@dataclass(frozen=True, slots=True)
class RiskSummary:
    equity_percentage: float
    fixed_income_percentage: float
    cash_percentage: float
    mixed_percentage: float
    other_percentage: float
    unclassified_percentage: float
    risk_band: str | None
    classified: list[ClassifiedAmount] = field(default_factory=list)


def classify_label(label: str) -> AssetClass:
    """Classify one holding/category name by keyword. Never guesses: unmatched -> unclassified."""
    for asset_class, pattern in _CLASSIFICATION_RULES:
        if pattern.search(label):
            return asset_class
    return "unclassified"


def classify_amounts(labeled_values: list[tuple[str, float]]) -> list[ClassifiedAmount]:
    return [
        ClassifiedAmount(label=label, asset_class=classify_label(label), value=value)
        for label, value in labeled_values
    ]


def summarize(amounts: list[ClassifiedAmount]) -> RiskSummary:
    """Aggregate classified amounts into percentages and, if enough is classified, a risk band."""
    total = sum(item.value for item in amounts)
    if total <= 0:
        return RiskSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, risk_band=None, classified=list(amounts))

    totals: dict[AssetClass, float] = dict.fromkeys(
        ("equity", "fixed_income", "cash", "mixed", "other", "unclassified"), 0.0
    )
    for item in amounts:
        totals[item.asset_class] += item.value
    pct = {asset_class: value / total * 100 for asset_class, value in totals.items()}

    classified_share = 1 - (pct["unclassified"] + pct["other"]) / 100
    risk_band = None
    if classified_share >= MIN_CLASSIFIED_SHARE_FOR_BAND:
        equity_like = pct["equity"] + pct["mixed"] * MIXED_EQUITY_WEIGHT
        for name, low, high in RISK_BANDS:
            if low <= equity_like < high:
                risk_band = name
                break

    return RiskSummary(
        equity_percentage=pct["equity"],
        fixed_income_percentage=pct["fixed_income"],
        cash_percentage=pct["cash"],
        mixed_percentage=pct["mixed"],
        other_percentage=pct["other"],
        unclassified_percentage=pct["unclassified"],
        risk_band=risk_band,
        classified=list(amounts),
    )
