from __future__ import annotations

import io
import posixpath
import zipfile
from urllib.parse import urljoin, urlparse

import requests

from pensionpeek.efast import USER_AGENT
from pensionpeek.models import Filing, RetrievedFiling

CURRENT_FILING_BASE = "https://efast2-filings-public.s3.amazonaws.com/prd/"
MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024
ALLOWED_DOWNLOAD_HOSTS = {
    "efast2-filings-public.s3.amazonaws.com",
    "www.efast.dol.gov",
    "efast.dol.gov",
}


class RetrievalError(RuntimeError):
    """Raised when a public filing is absent, excluded, or temporarily unavailable."""


def resolve_filing_url(pdf_path: str) -> str:
    path = pdf_path.strip()
    if not path:
        raise RetrievalError("This search result does not include a public filing path.")
    if path.startswith(("http://", "https://")):
        candidate = path
    else:
        candidate = urljoin(CURRENT_FILING_BASE, path.lstrip("/"))
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise RetrievalError("The filing URL points outside an approved DOL disclosure host.")
    return candidate


def _pdf_from_archive(content: bytes) -> tuple[bytes, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            candidates = [
                info
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".pdf")
            ]
            if not candidates:
                raise RetrievalError("The filing archive did not contain a PDF.")
            chosen = max(candidates, key=lambda info: info.file_size)
            if chosen.file_size > MAX_DOWNLOAD_BYTES:
                raise RetrievalError(
                    "The PDF inside this filing archive is too large to process safely."
                )
            filename = posixpath.basename(chosen.filename) or "form-5500.pdf"
            return archive.read(chosen), filename
    except zipfile.BadZipFile as exc:
        raise RetrievalError(
            "The filing download was neither a readable PDF nor a ZIP archive."
        ) from exc


def _read_limited(response: requests.Response) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
        raise RetrievalError("This filing is too large to process in the app (80 MB limit).")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=128 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_DOWNLOAD_BYTES:
            raise RetrievalError("This filing is too large to process in the app (80 MB limit).")
        chunks.append(chunk)
    return b"".join(chunks)


def download_filing(
    filing: Filing,
    session: requests.Session | None = None,
) -> RetrievedFiling:
    url = resolve_filing_url(filing.pdf_path)
    client = session or requests.Session()
    try:
        response = client.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/pdf, application/zip, */*"},
            timeout=(8, 60),
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
        final_host = urlparse(response.url).hostname
        if final_host not in ALLOWED_DOWNLOAD_HOSTS:
            raise RetrievalError("The filing download redirected outside an approved DOL host.")
        content = _read_limited(response)
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        if status in {403, 404}:
            message = (
                "DOL does not expose a downloadable image for this filing. Some older, foreign, "
                "one-participant, amended, or sensitive filings are not publicly disclosed."
            )
        else:
            message = "The public filing could not be downloaded from DOL right now."
        raise RetrievalError(message) from exc

    if content.startswith(b"%PDF-"):
        filename = posixpath.basename(urlparse(url).path) or f"{filing.key}.pdf"
        return RetrievedFiling(content, filename, url, was_archive=False)
    if content.startswith(b"PK"):
        pdf_bytes, filename = _pdf_from_archive(content)
        if not pdf_bytes.startswith(b"%PDF-"):
            raise RetrievalError("The archive's PDF entry is not a valid PDF file.")
        return RetrievedFiling(pdf_bytes, filename, url, was_archive=True)

    raise RetrievalError(
        "DOL returned a web page instead of the filing, so PensionPeek could not parse it."
    )
