from __future__ import annotations

import io
import zipfile

import pytest

from planpeek.retrieval import RetrievalError, _pdf_from_archive, resolve_filing_url


def test_resolves_current_public_s3_path() -> None:
    url = resolve_filing_url("/2020/06/04/example.pdf")
    assert url == "https://efast2-filings-public.s3.amazonaws.com/prd/2020/06/04/example.pdf"


def test_rejects_untrusted_filing_host() -> None:
    with pytest.raises(RetrievalError, match="approved"):
        resolve_filing_url("https://example.com/not-a-filing.pdf")


def test_extracts_largest_pdf_from_zip() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "hello")
        archive.writestr("small.pdf", b"%PDF-small")
        archive.writestr("folder/main.pdf", b"%PDF-main filing contents")
    content, filename = _pdf_from_archive(buffer.getvalue())
    assert content == b"%PDF-main filing contents"
    assert filename == "main.pdf"
