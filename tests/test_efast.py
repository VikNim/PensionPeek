from __future__ import annotations

import pytest

from planpeek.efast import build_query, normalize_ein
from planpeek.models import Filing


def test_normalize_ein_accepts_display_format() -> None:
    assert normalize_ein("77-0493581") == "770493581"


def test_normalize_ein_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="9 digits"):
        normalize_ein("1234")


def test_company_query_searches_plan_and_sponsor() -> None:
    query = build_query('Acme "North"', "Company")
    assert 'planname:"Acme \\"North\\""' in query
    assert 'plansponsor:"Acme \\"North\\""' in query


def test_filing_maps_public_search_hit() -> None:
    filing = Filing.from_hit(
        {
            "id": "abc",
            "fields": {
                "planname": "ACME 401(K)",
                "plansponsor": "ACME LLC",
                "ein": "123456789",
                "pn": "001",
                "planyear": "2024",
                "participantsboy": "42",
                "assetseoy": "1234567.0",
                "pdfpath": "/2025/example.pdf",
            },
        }
    )
    assert filing.formatted_ein == "12-3456789"
    assert filing.plan_year == 2024
    assert filing.participants_boy == 42
    assert filing.assets_eoy == 1_234_567
