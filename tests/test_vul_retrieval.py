from __future__ import annotations

import pytest
import requests

from pensionpeek.models import VulFiling
from pensionpeek.vul_retrieval import (
    VulRetrievalError,
    download_vul_document,
    resolve_document_url,
)


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


class _FakeResponse:
    def __init__(self, content: bytes, url: str, status: int = 200) -> None:
        self._content = content
        self.url = url
        self.status_code = status
        self.headers: dict = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code}")
            error.response = self
            raise error

    def iter_content(self, chunk_size: int):
        yield self._content


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def get(self, url, headers=None, timeout=None, allow_redirects=None, stream=None):
        return self.response


def test_resolve_document_url_builds_archive_path() -> None:
    url = resolve_document_url(_filing())
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/1048607/000072686526000669/initialn6.htm"
    )


def test_resolve_document_url_rejects_missing_fields() -> None:
    with pytest.raises(VulRetrievalError, match="does not include a document"):
        resolve_document_url(_filing(document_filename=""))


def test_download_vul_document_decodes_html() -> None:
    filing = _filing()
    url = resolve_document_url(filing)
    session = _FakeSession(_FakeResponse(b"<html>hello</html>", url))

    retrieved = download_vul_document(filing, session=session)

    assert retrieved.html == "<html>hello</html>"
    assert retrieved.filename == "initialn6.htm"
    assert retrieved.source_url == url


def test_download_vul_document_rejects_redirect_outside_sec() -> None:
    filing = _filing()
    session = _FakeSession(
        _FakeResponse(b"<html></html>", "https://not-sec.example.com/initialn6.htm")
    )
    with pytest.raises(VulRetrievalError, match="approved SEC host"):
        download_vul_document(filing, session=session)


def test_download_vul_document_surfaces_not_found() -> None:
    filing = _filing()
    url = resolve_document_url(filing)
    session = _FakeSession(_FakeResponse(b"", url, status=404))
    with pytest.raises(VulRetrievalError, match="does not expose this document"):
        download_vul_document(filing, session=session)


def test_download_vul_document_403_points_at_user_agent_requirement() -> None:
    # Confirmed against a live filing: SEC returns 403, not 404, when EDGAR_USER_AGENT
    # isn't a compliant contact string -- this must not be reported as "not found."
    filing = _filing()
    url = resolve_document_url(filing)
    session = _FakeSession(_FakeResponse(b"", url, status=403))
    with pytest.raises(VulRetrievalError, match="EDGAR_USER_AGENT"):
        download_vul_document(filing, session=session)
