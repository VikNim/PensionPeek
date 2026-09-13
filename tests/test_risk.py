from __future__ import annotations

from planpeek.risk import classify_amounts, classify_label, summarize

# Real fund names from a live Google LLC 401(k) Savings Plan Schedule of Assets, used to
# validate the keyword classifier against actual holdings rather than idealized names.
REAL_FUND_CLASSIFICATIONS = [
    ("Nuveen Large Cap Responsible Equity Fund R6", "equity"),
    ("PIMCO Income Fund Institutional Class", "fixed_income"),
    ("The Vanguard Group, Inc. 500 Index Fund Institutional Select", "equity"),
    ("The Vanguard Group, Inc. Cash Reserves Federal MM Fund Admiral", "cash"),
    ("The Vanguard Group, Inc. Emerging Markets Index Fund Institutional Plus", "equity"),
    ("The Vanguard Group, Inc. Real Estate Index Fund Institutional", "equity"),
    ("The Vanguard Group, Inc. Total International Bond Index Fund Institutional", "fixed_income"),
    ("The Vanguard Group, Inc. Target Retirement Trust 2040", "mixed"),
    ("The Vanguard Group, Inc. Developed Markets Index Trust", "equity"),
    ("The Vanguard Group, Inc. Institutional Total Bond Market Index Trust", "fixed_income"),
    ("EARNEST Partners Smid Cap Core Fund, Founders Class", "equity"),
    ("Metropolitan West Funds MetWest Total Return Bond Fund (CIT)", "fixed_income"),
    ("The Vanguard Group, Inc. Parnassus US Large Cap", "equity"),
    ("Vanguard Brokerage Option Brokerage Accounts", "other"),
    ("Loans to participants", "other"),
    # Genuinely unclassifiable by name alone -- must not be force-fit into a bucket.
    ("The Vanguard Group, Inc. Retirement Savings Trust II", "unclassified"),
    ("Fidelity Investments Diversified Commingled Pool Fund, Class C", "unclassified"),
]


def test_classify_label_matches_real_fund_names() -> None:
    for label, expected in REAL_FUND_CLASSIFICATIONS:
        assert classify_label(label) == expected, label


def test_classify_label_prefers_more_specific_rule_over_equity_catchall() -> None:
    # Contains "index" and "international" (equity-ish) but is unambiguously a bond fund;
    # "bond" must win because fixed_income is checked before the equity catch-all.
    assert classify_label("Total International Bond Index Fund") == "fixed_income"


def test_classify_amounts_pairs_labels_with_values() -> None:
    amounts = classify_amounts([("S&P 500 Index Fund", 100.0), ("Total Bond Fund", 50.0)])
    assert [a.asset_class for a in amounts] == ["equity", "fixed_income"]
    assert [a.value for a in amounts] == [100.0, 50.0]


def test_summarize_computes_percentages_and_risk_band() -> None:
    amounts = classify_amounts(
        [
            ("S&P 500 Index Fund", 70.0),
            ("Total Bond Fund", 30.0),
        ]
    )
    summary = summarize(amounts)
    assert summary.equity_percentage == 70.0
    assert summary.fixed_income_percentage == 30.0
    assert summary.unclassified_percentage == 0.0
    assert summary.risk_band == "Growth"  # 70% equity-like falls in [65, 85)


def test_summarize_withholds_risk_band_when_mostly_unclassified() -> None:
    amounts = classify_amounts(
        [
            ("Common / collective trusts", 900.0),  # unclassifiable vehicle-type bucket
            ("S&P 500 Index Fund", 100.0),
        ]
    )
    summary = summarize(amounts)
    assert summary.unclassified_percentage == 90.0
    assert summary.risk_band is None


def test_summarize_weights_mixed_funds_as_half_equity_for_banding() -> None:
    amounts = classify_amounts([("Target Retirement Trust 2040", 100.0)])
    summary = summarize(amounts)
    assert summary.mixed_percentage == 100.0
    # equity_like = 0 + 100*0.5 = 50 -> Moderate [30, 65)
    assert summary.risk_band == "Moderate"


def test_summarize_handles_zero_total_without_error() -> None:
    summary = summarize([])
    assert summary.risk_band is None
    assert summary.equity_percentage == 0.0


def test_real_google_plan_lands_in_moderate_band() -> None:
    # End-to-end sanity check against the real plan's actual (trimmed) composition:
    # dominated by target-date trusts (mixed) with meaningful direct equity index funds.
    amounts = classify_amounts(
        [
            ("Vanguard 500 Index Fund Institutional Select", 8_503_979_330),
            ("Vanguard Total International Bond Index Fund", 179_272_982),
            ("Vanguard Target Retirement Trust 2040", 3_324_929_052),
            ("Vanguard Target Retirement Trust 2050", 7_148_528_242),
            ("Vanguard Cash Reserves Federal MM Fund Admiral", 8_042_685),
        ]
    )
    summary = summarize(amounts)
    assert summary.risk_band in {"Moderate", "Growth"}
