from __future__ import annotations

from planpeek.schedule_of_assets import extract_holdings

# A real Schedule H, Line 4i (Schedule of Assets) page, captured from a live Google LLC
# 401(k) Savings Plan Form 5500 filing (filing_id 20250827123612NAL0005026227001, plan
# year 2024). Used verbatim so extraction is validated against real DOL PDF-extraction
# messiness, not an idealized fixture.
REAL_GOOGLE_SCHEDULE_OF_ASSETS = """
GOOGLE LLC 401(k) SAVINGS PLAN
         SCHEDULE H, LINE 4i – SCHEDULE OF ASSETS (HELD AT END OF YEAR)
                                 December 31, 2024


Plan Sponsor:                           Google LLC
Employer Identification Number:            77-0493581
Plan Number:                              001


                   (b)                                                  (c)
                Identity of                        Description of Investment, Including                                        (e)
            Issue, Lessor                         Maturity Date, Rate of Interest,                          (d)         Current
 (a)       or Similar Party                         Collateral, Par or Maturity Value                   Cost         Value

    Mutual Funds
    Nuveen                               Large Cap Responsible Equity Fund R6                    **   $      273,217,178
   PIMCO                             Income Fund Institutional Class                              **             2,918,233
  *  The Vanguard Group, Inc.              500 Index Fund Institutional Select                          **         8,503,979,330
  *  The Vanguard Group, Inc.             Cash Reserves Federal MM Fund Admiral                 **             8,042,685
  *  The Vanguard Group, Inc.              Emerging Markets Index Fund Institutional Plus            **          464,235,808
  *  The Vanguard Group, Inc.               Real Estate Index Fund Institutional                         **          370,472,575
  *  The Vanguard Group, Inc.                 Total International Bond Index Fund Institutional           **          179,272,982
  *  The Vanguard Group, Inc.                Wellesley Income Fund Admiral                             **          519,010,734

     Collective Trusts
  *  The Vanguard Group, Inc.                Target Retirement Trust 2020                                **          114,140,981
  *  The Vanguard Group, Inc.                Target Retirement Trust 2025                                **          378,777,049
  *  The Vanguard Group, Inc.               Retirement Savings Trust II                                  **          400,434,346
      Fidelity Investments                        Diversified Commingled Pool Fund, Class C               **          210,007,040
   EARNEST Partners                  Smid Cap Core Fund, Founders Class                      **          264,305,773
     Metropolitan West Funds              MetWest Total Return Bond Fund (CIT)                     **            87,955,982

    Custom Investment Trust
  *  The Vanguard Group, Inc.              Parnassus US Large Cap                                    **          785,845,403

     Self-Directed Brokerage Accounts
  *  Vanguard Brokerage Option                         Brokerage Accounts                           **         1,891,033,914    Notes Receivable from Participants     Self‐Directed
  *  Loans to participants                        Interest rates ranging from 4.00% to 9.50%
                                                with maturities from January 2025
                                                    to August 2035                                               **          120,942,045

     Total                                                                                 $   10,000,000,000


* Party-in-interest.
** Cost information in column (d) has been omitted as all investments are participant directed.




                           See Independent Auditor’s Report.

                                                                                                   12.
"""


def test_extract_holdings_returns_empty_when_schedule_absent() -> None:
    holdings, warnings = extract_holdings("This filing has no supplemental schedule at all.")
    assert holdings == []
    assert warnings == []


def test_extract_holdings_parses_real_filing_text_cleanly() -> None:
    holdings, warnings = extract_holdings(REAL_GOOGLE_SCHEDULE_OF_ASSETS)

    identities = [h.identity for h in holdings]
    # The header row's own column-label text ("Identity of ... Current Value") must not
    # leak into the first real holding's identity.
    assert not any("Description of Investment" in identity for identity in identities)
    assert not any("Par or Maturity Value" in identity for identity in identities)

    by_fragment = {name: h.value for h in holdings for name in [h.identity] if "500 Index" in name}
    assert by_fragment
    assert next(iter(by_fragment.values())) == 8_503_979_330

    nuveen = next(h for h in holdings if "Nuveen" in h.identity)
    assert nuveen.value == 273_217_178
    assert "Large Cap Responsible Equity Fund" in nuveen.identity


def test_extract_holdings_stops_at_total_line_and_skips_footnotes() -> None:
    holdings, _warnings = extract_holdings(REAL_GOOGLE_SCHEDULE_OF_ASSETS)
    assert not any(h.identity.strip().lower() == "total" for h in holdings)
    # Footnote lines ("* Party-in-interest.") must never surface as fake holdings.
    assert not any("party-in-interest" in h.identity.lower() for h in holdings)


def test_extract_holdings_warns_when_sum_does_not_reconcile_with_total() -> None:
    # The fixture's Total line (10,000,000,000) is deliberately set well below the real
    # sum of its own listed rows (~12.68B), mirroring the kind of reconciliation gap a
    # multi-line wrapped description (brokerage/loan rows) can cause in the real filing
    # this fixture is based on -- confirms the drift check fires rather than silently
    # under-reporting.
    holdings, warnings = extract_holdings(REAL_GOOGLE_SCHEDULE_OF_ASSETS)
    extracted_total = sum(h.value for h in holdings)
    assert extracted_total > 12_000_000_000
    assert len(warnings) == 1
    assert "doesn't reconcile" in warnings[0]


def test_extract_holdings_reconciles_cleanly_when_total_matches() -> None:
    text = """
    SCHEDULE H, LINE 4i - SCHEDULE OF ASSETS (HELD AT END OF YEAR)
    Identity of Issue      Description of Investment      Cost   Current Value
    Vanguard    500 Index Fund   **   1,000,000
    Vanguard    Total Bond Market Index Fund  **  500,000
    Total                                             $  1,500,000
    """
    holdings, warnings = extract_holdings(text)
    assert len(holdings) == 2
    assert warnings == []
