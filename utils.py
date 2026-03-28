"""
Shared utilities for Pocket TTS Studio.
"""

import re
import requests
from bs4 import BeautifulSoup

MAX_CHARS_PER_CHUNK = 500
VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "aside",
               "form", "button", "noscript", "iframe"]


def split_into_sentences(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split text into sentence-sized chunks for TTS processing."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for sent in raw:
        if len(buf) + len(sent) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += " " + sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def extract_text_from_url(url: str, timeout: int = 30) -> str:
    """Fetch a public URL and extract the main text content."""
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    container = soup.find("article") or soup.find("main") or soup.find("body")
    if container is None:
        raise ValueError("Could not extract text from the page.")

    paragraphs = container.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    if not text:
        text = container.get_text(separator="\n", strip=True)
    return text
