"""Web page fetching and text extraction (HTML -> plain text)."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 15
_USER_AGENT = (
    "Mozilla/5.0 (compatible; ChemSynthAssistant/1.0; "
    "+https://github.com/example/chemsynthassistant)"
)
_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TOOL_NAME = "ChemSynthAssistant"
_TOOL_EMAIL = "chemsynthassistant@example.com"

_MAX_TEXT_LENGTH = 15_000


def fetch_page(url: str, *, timeout: int = _REQUEST_TIMEOUT) -> str | None:
    """Download an HTML page and return the raw HTML string."""
    try:
        resp = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
            allow_redirects=True,
        )
        if resp.status_code == 200:
            return resp.text
        logger.warning("fetch_page %s returned %s", url, resp.status_code)
        return None
    except requests.RequestException as exc:
        logger.warning("fetch_page failed (%s): %s", url, exc)
        return None


def extract_text(html: str) -> str:
    """Strip HTML tags, scripts, and styles — return clean plain text."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed; returning raw HTML slice")
        return html[:_MAX_TEXT_LENGTH]

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clean = "\n".join(lines)
    return clean[:_MAX_TEXT_LENGTH]


def extract_pubmed_abstract(pmid: str | int) -> str | None:
    """Fetch the abstract for a PubMed article via E-utilities efetch (XML)."""
    url = (
        f"{_EUTILS_BASE}/efetch.fcgi?"
        f"db=pubmed&id={pmid}&rettype=abstract&retmode=xml"
        f"&tool={_TOOL_NAME}&email={_TOOL_EMAIL}"
    )
    try:
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return None
    except requests.RequestException as exc:
        logger.warning("efetch failed for PMID %s: %s", pmid, exc)
        return None

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(resp.text)
        abstract_parts: list[str] = []
        for elem in root.iter("AbstractText"):
            label = elem.get("Label", "")
            text = "".join(elem.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)

        title_elem = root.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

        parts: list[str] = []
        if title:
            parts.append(title)
        if abstract_parts:
            parts.append("\n".join(abstract_parts))
        return "\n\n".join(parts) if parts else None

    except Exception as exc:
        logger.warning("XML parse error for PMID %s: %s", pmid, exc)
        return None


def fetch_and_extract(url: str) -> str | None:
    """Convenience: fetch a page and extract its text in one call."""
    html = fetch_page(url)
    if html is None:
        return None
    return extract_text(html)
