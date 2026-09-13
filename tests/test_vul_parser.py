from __future__ import annotations

import json

import pytest

from planpeek.models import VulFiling
from planpeek.vul_parser import (
    VulParsingError,
    _extract_tables,
    _is_fund_menu_shaped,
    _parse_llm_json,
    extract_vul_filing,
    relevant_table_text,
)

# Mimics real N-6 markup: fund name and expense ratio each buried under nested styling
# divs/fonts inside their own <td>, plus a small boilerplate table (table of contents)
# that must NOT be picked up as a fund menu.
FUND_TABLE_HTML = """
<html><body>
<table>
  <tr><td>Contents</td><td>Page</td></tr>
  <tr><td>Fee Table</td><td>12</td></tr>
  <tr><td>Principal Risks</td><td>15</td></tr>
</table>
<table>
  <tr><td>Fund</td><td>Expenses</td><td>1yr</td><td>5yr</td></tr>
  <tr>
    <td><div><font>Fidelity</font><font>&reg;</font><font> VIP Contrafund</font></div></td>
    <td><div><font>0.54%</font></div></td>
    <td><div><font>21.52%</font></div></td>
    <td><div><font>15.37%</font></div></td>
  </tr>
  <tr>
    <td><div><font>Vanguard VIF Growth</font></div></td>
    <td><div><font>0.35%</font></div></td>
    <td><div><font>18.20%</font></div></td>
    <td><div><font>13.10%</font></div></td>
  </tr>
  <tr>
    <td>Nested layout <table><tr><td>PIMCO Total Return</td><td>0.60%</td></tr></table></td>
    <td></td><td></td><td></td>
  </tr>
</table>
</body></html>
"""


def _filing(**overrides) -> VulFiling:
    defaults = dict(
        cik="1048607",
        accession_number="0000726865-26-000669",
        form_type="N-6",
        file_date="2026-08-06",
        display_name="Lincoln Life Flexible Premium Variable Life Account M",
        document_filename="initialn6.htm",
    )
    defaults.update(overrides)
    return VulFiling(**defaults)


def test_extract_tables_flattens_nested_tables_separately() -> None:
    tables = _extract_tables(FUND_TABLE_HTML)
    # A nested <table> closes before its parent, so it's collected first: table of
    # contents, then the nested layout table, then the outer fund menu table last.
    assert len(tables) == 3
    nested = tables[1]
    assert nested == [["PIMCO Total Return", "0.60%"]]


def test_is_fund_menu_shaped_rejects_boilerplate_table() -> None:
    tables = _extract_tables(FUND_TABLE_HTML)
    toc_table = tables[0]
    assert _is_fund_menu_shaped(toc_table) is False


def test_is_fund_menu_shaped_accepts_percentage_dense_table() -> None:
    tables = _extract_tables(FUND_TABLE_HTML)
    fund_table = tables[-1]
    assert _is_fund_menu_shaped(fund_table) is True


def test_relevant_table_text_includes_fund_rows_not_boilerplate() -> None:
    text = relevant_table_text(FUND_TABLE_HTML)
    assert "Fidelity" in text
    assert "0.54%" in text
    assert "Vanguard VIF Growth" in text
    assert "Principal Risks" not in text


def test_relevant_table_text_raises_when_nothing_fund_shaped() -> None:
    with pytest.raises(VulParsingError, match="No fund or asset-allocation table"):
        relevant_table_text("<html><table><tr><td>hello</td></tr></table></html>")


def test_relevant_table_text_truncates_at_max_chars() -> None:
    text = relevant_table_text(FUND_TABLE_HTML, max_chars=40)
    assert len(text) < 200
    assert "truncated" in text


def test_parse_llm_json_strips_markdown_fence() -> None:
    fenced = '```json\n{"a": 1}\n```'
    assert _parse_llm_json(fenced) == {"a": 1}


def test_parse_llm_json_raises_on_garbage() -> None:
    with pytest.raises(VulParsingError, match="not return valid JSON"):
        _parse_llm_json("not json at all")


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.messages = None

    def invoke(self, messages):
        self.messages = messages

        class _Response:
            content = json.dumps(self.payload)

        return _Response()


def test_extract_vul_filing_end_to_end_with_fake_llm() -> None:
    payload = {
        "policy_investment_type": "Variable",
        "carrier_name": "Lincoln National",
        "reporting_year": 2026,
        "allocation_chart": [{"asset_category": "Equity", "percentage": 98.0}],
        "unclassified_percentage": 2.0,
        "unclassified_reason": "rounding",
        "risk_metrics": {"equity_exposure_percentage": 98.0},
        "sub_funds_list": [
            {
                "name": "Fidelity VIP Contrafund",
                "manager": "Fidelity",
                "allocation_weight": 25.0,
                "expense_ratio": 0.54,
            }
        ],
        "warnings": [],
    }
    llm = _FakeLLM(payload)

    result = extract_vul_filing(FUND_TABLE_HTML, _filing(), llm=llm)

    assert result.carrier_name == "Lincoln National"
    assert result.reporting_year == 2026
    assert result.allocation_chart[0].asset_category == "Equity"
    assert result.unclassified_reason == "rounding"
    assert result.sub_funds_list[0].name == "Fidelity VIP Contrafund"
    assert result.sub_funds_list[0].expense_ratio == 0.54
    # The narrowed table text (not the full raw HTML) was sent to the model.
    sent_text = llm.messages[1].content
    assert "Fidelity" in sent_text
    assert "Principal Risks" not in sent_text


def test_extract_vul_filing_raises_on_invalid_model_json() -> None:
    class _BadLLM:
        def invoke(self, messages):
            class _Response:
                content = "not json"

            return _Response()

    with pytest.raises(VulParsingError, match="not return valid JSON"):
        extract_vul_filing(FUND_TABLE_HTML, _filing(), llm=_BadLLM())
