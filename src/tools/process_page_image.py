"""process_page_image tool: vision-based QA over PDF pages.

For each requested page:
  1. Check the local PNG cache; render via PyMuPDF only on a cache miss.
  2. Base64-encode all page images.
  3. Send a single multimodal request (all pages + question) to the Azure
     OpenAI vision model and return the text response.

Cache location: <project_root>/embeddings/officeqa/page_images/<pdf_stem>/
Cache filename: <pdf_stem>_page_<N>.png  (N = 1-indexed page number)
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Optional

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.tools import tool

_PAGE_IMAGE_CACHE_DIR = _ROOT / "embeddings" / "officeqa" / "page_images"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(pdf_stem: str, page_id: int) -> Path:
    """Return the expected cache path for one page image."""
    return _PAGE_IMAGE_CACHE_DIR / pdf_stem / f"{pdf_stem}_page_{page_id}.png"


def _render_page(pdf_path: str, page_id: int, dpi_scale: float = 2.5) -> bytes:
    """Render a single PDF page (1-indexed) to PNG bytes using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    if page_id < 1 or page_id > doc.page_count:
        raise ValueError(
            f"Page {page_id} out of range (document has {doc.page_count} pages)"
        )
    page = doc[page_id - 1]
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _load_or_render(pdf_path: str, pdf_stem: str, page_id: int) -> bytes:
    """Return PNG bytes for page_id, using cache when available."""
    cache_file = _cache_path(pdf_stem, page_id)
    if cache_file.is_file():
        return cache_file.read_bytes()

    png_bytes = _render_page(pdf_path, page_id)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(png_bytes)
    return png_bytes


def _call_vision(images: list[bytes], question: str) -> str:
    """Send all page images plus the question to the Azure vision model."""
    from openai import AzureOpenAI
    from azure_local import load_azure_credentials_from_local

    api_key, api_version, endpoint, deployment = load_azure_credentials_from_local()
    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    content: list[dict] = []
    for img_bytes in images:
        b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": question})

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=2000,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def process_page_image(
    path: str = "",
    page_ids: Optional[list] = None,
    question: str = "",
) -> str:
    """Answer a question by visually analysing one or more PDF pages.

    Args:
        path:     Absolute or relative path to the PDF file.
        page_ids: List of 1-indexed page numbers to include (e.g. [1, 2]).
        question: The question to answer based on the page images.

    Returns:
        The vision model's text answer, or an error string.

    Images are cached at
    ``embeddings/officeqa/page_images/<pdf_stem>/<pdf_stem>_page_<N>.png``
    so that re-runs skip rendering for pages already processed.
    """
    p = (path or "").strip()
    if not p:
        return "(no path provided)"

    fp = Path(p)
    if not fp.is_file():
        return f"(file not found: {p})"

    ids = page_ids or []
    if not ids:
        return "(no page_ids provided)"

    q = (question or "").strip()
    if not q:
        return "(no question provided)"

    pdf_stem = fp.stem
    images: list[bytes] = []
    error_notes: list[str] = []

    for pid in ids:
        try:
            png = _load_or_render(str(fp.resolve()), pdf_stem, int(pid))
            images.append(png)
        except ValueError as exc:
            error_notes.append(f"page {pid}: {exc}")
        except Exception as exc:
            error_notes.append(f"page {pid}: render error — {exc}")

    if not images:
        note = "; ".join(error_notes) if error_notes else "all pages failed"
        return f"(no valid pages could be loaded: {note})"

    try:
        answer = _call_vision(images, q)
    except Exception as exc:
        return f"(vision API error: {exc})"

    if error_notes:
        answer += "\n\n[Skipped pages: " + "; ".join(error_notes) + "]"
    return answer
