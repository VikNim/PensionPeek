"""SEC EDGAR document retrieval for VUL filings.

Mirrors pensionpeek.retrieval's shape (resolve -> validate host -> download, size-capped)
for SEC's document archive instead of DOL's S3 bucket.
"""

from __future__ import annotations

from urllib.parse import urlparse

import requests

from pensionpeek.edgar import edgar_user_agent
from pensionpeek.models import RetrievedVulDocument, VulFiling

ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_DOWNLOAD_HOSTS = {"www.sec.gov"}


class VulRetrievalError(RuntimeError):
    """Raised when a VUL filing document is absent or cannot be downloaded."""


def resolve_document_url(filing: VulFiling) -> str:
    if not filing.cik or not filing.accession_number or not filing.document_filename:
        raise VulRetrievalError("This filing result does not include a document to fetch.")
    candidate = (
        f"{ARCHIVE_BASE}/{filing.cik}/{filing.accession_no_dashes}/{filing.document_filename}"
    )
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise VulRetrievalError("The filing document URL points outside an approved SEC host.")
    return candidate


def _read_limited(response: requests.Response) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
        raise VulRetrievalError("This filing document is too large to process safely.")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=128 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_DOWNLOAD_BYTES:
            raise VulRetrievalError("This filing document is too large to process safely.")
        chunks.append(chunk)
    return b"".join(chunks)


def download_vul_document(
    filing: VulFiling,
    session: requests.Session | None = None,
) -> RetrievedVulDocument:
    url = resolve_document_url(filing)
    client = session or requests.Session()
    try:
        response = client.get(
            url,
            headers={"User-Agent": edgar_user_agent(), "Accept": "text/html, */*"},
            timeout=(8, 60),
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
        final_host = urlparse(response.url).hostname
        if final_host not in ALLOWED_DOWNLOAD_HOSTS:
            raise VulRetrievalError("The filing document redirected outside an approved SEC host.")
        content = _read_limited(response)
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 403:
            # Confirmed against a real, currently-public filing: SEC returns 403 here (not
            # 404) when EDGAR_USER_AGENT isn't a compliant "Name contact@domain" identifier,
            # even though the same placeholder header works fine against the JSON APIs.
            message = (
                "SEC EDGAR rejected this request (403). Set EDGAR_USER_AGENT to a real "
                "'Your Company Name admin@yourcompany.com' identifier -- see "
                "https://www.sec.gov/os/webmaster-faq#developers -- and try again."
            )
        elif status == 404:
            message = (
                "SEC EDGAR does not expose this document. It may have been withdrawn or "
                "the accession number may be stale."
            )
        else:
            message = "The filing document could not be downloaded from SEC EDGAR right now."
        raise VulRetrievalError(message) from exc

    try:
        html = content.decode("utf-8")
    except UnicodeDecodeError:
        html = content.decode("latin-1")
    return RetrievedVulDocument(html=html, filename=filing.document_filename, source_url=url)
