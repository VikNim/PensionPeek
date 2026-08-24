from __future__ import annotations

from pensionpeek.parser import chunk_pages, extract_metrics_from_text

SAMPLE_TEXT = """
Form 5500 Annual Return/Report
Total number of participants at the beginning of the plan year 42
Total number of participants at the end of the plan year 48

Schedule H Financial Information
Total assets 1,000,000 1,250,000
Total liabilities 10,000 20,000
Net assets available for benefits 990,000 1,230,000
Interest-bearing cash 100,000 125,000
Pooled separate accounts 300,000 350,000
Total income 200,000
Total expenses 150,000
"""


def test_extracts_schedule_h_metrics() -> None:
    metrics = extract_metrics_from_text(SAMPLE_TEXT)
    assert metrics.assets_boy == 1_000_000
    assert metrics.assets_eoy == 1_250_000
    assert metrics.liabilities_eoy == 20_000
    assert metrics.net_assets_eoy == 1_230_000
    assert metrics.total_income == 200_000
    assert metrics.total_expenses == 150_000
    assert metrics.net_income == 50_000
    assert metrics.participants_boy == 42
    assert metrics.participants_eoy == 48
    assert metrics.asset_categories["Interest-bearing cash"] == 125_000


def test_chunks_keep_page_provenance_and_overlap() -> None:
    chunks = chunk_pages(["A sentence. " * 300, "Second page."], chunk_size=300, overlap=50)
    assert len(chunks) > 2
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    assert chunks[-1].chunk_id.startswith("p2-")


def test_empty_pages_are_skipped() -> None:
    chunks = chunk_pages(["", "Useful filing text"])
    assert len(chunks) == 1
    assert chunks[0].page == 2


def test_chunks_exclude_masked_form_artifacts_from_llm_context() -> None:
    chunks = chunk_pages(["Sponsor ABCDEFGHI\nAssets -123456789012345\nVisible amount $42,000"])

    assert len(chunks) == 1
    assert "ABCDEFGHI" not in chunks[0].text
    assert "-123456789012345" not in chunks[0].text
    assert "$42,000" in chunks[0].text


def test_non_schedule_h_does_not_invent_financials_from_nearby_identifiers() -> None:
    text = """
    Form 5500-SF
    Total number of participants at the beginning of the plan year 1234567811
    Total income 12345678901234580435
    Total expenses 12345678901234516364
    """
    metrics = extract_metrics_from_text(text)
    assert metrics.participants_boy == 11
    assert metrics.assets_eoy is None
    assert metrics.total_income is None
    assert metrics.total_expenses is None
    assert metrics.asset_categories == {}


def test_decodes_efast_hidden_numeric_sentinels() -> None:
    text = """
    Total number of participants at the beginning of the plan year 5a 1234567842
    Schedule H
    1f Total assets (add all amounts in lines 1a through 1e) 1f
    -12345678901234553225853429 -12345678901234565646011638
    2j Total expenses. Add all expense amounts in column (b) 2j -1234567890123452964377796
    """
    metrics = extract_metrics_from_text(text)
    assert metrics.participants_boy == 42
    assert metrics.assets_boy == 53_225_853_429
    assert metrics.assets_eoy == 65_646_011_638
    assert metrics.total_expenses == 2_964_377_796
