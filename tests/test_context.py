from __future__ import annotations

from planpeek.context import contains_mask_artifact, sanitize_filing_context


def test_removes_known_text_masks_and_empty_numeric_sentinels() -> None:
    raw = "Sponsor ABCDEFGHI\nMasked XXXXXXXXX\nAmount -123456789012345"

    cleaned = sanitize_filing_context(raw)

    assert cleaned == "Sponsor\nMasked\nAmount"
    assert not contains_mask_artifact(cleaned)


def test_preserves_visible_value_appended_to_efast_numeric_sentinel() -> None:
    raw = "Total assets -12345678901234567890"

    assert sanitize_filing_context(raw) == "Total assets 67890"
    assert contains_mask_artifact(raw)
