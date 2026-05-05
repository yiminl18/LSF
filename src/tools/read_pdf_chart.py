"""read_pdf_chart tool: render a PDF page and extract chart data via GPT vision.

Uses PyMuPDF to render a specific page of a PDF to a PNG image, then sends
it to the Azure OpenAI vision model to extract structured chart data.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.tools import tool


def _render_pdf_page(pdf_path: str, page_number: int, dpi_scale: float = 2.5) -> bytes:
    """Render a PDF page to PNG bytes using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > doc.page_count:
        raise ValueError(f"Page {page_number} out of range (1-{doc.page_count})")
    page = doc[page_number - 1]
    mat = fitz.Matrix(dpi_scale, dpi_scale)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _extract_embedded_images(pdf_path: str, page_number: int) -> list[bytes]:
    """Extract embedded raster images from a specific PDF page."""
    import fitz

    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    results = []
    for img_info in page.get_images():
        xref = img_info[0]
        base_image = doc.extract_image(xref)
        results.append(base_image["image"])
    return results


def _ask_vision(image_bytes: bytes, prompt: str) -> str:
    """Send an image to the Azure OpenAI vision model and return the response."""
    from openai import AzureOpenAI
    from azure_local import load_azure_credentials_from_local

    api_key, api_version, endpoint, deployment = load_azure_credentials_from_local()
    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

    img_b64 = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_completion_tokens=1000,
    )
    return response.choices[0].message.content or ""


@tool
def read_pdf_chart(
    path: str = "",
    page_number: int = 1,
    chart_description: str = "",
    use_embedded: bool = True,
) -> str:
    """Extract chart data from a PDF page using vision analysis.

    Renders the specified ``page_number`` (1-indexed) of the PDF at ``path``
    and asks the vision model to read the chart values.  If ``use_embedded``
    is True (default), also tries to extract any embedded raster image on
    that page (which is often the actual chart bitmap) and sends that
    separately for higher accuracy.

    ``chart_description`` helps the model identify which chart to read if
    multiple charts are present.

    Returns a text description of the chart values extracted by the vision
    model.
    """
    p = (path or "").strip()
    if not p:
        return "(no path provided)"
    fp = Path(p)
    if not fp.is_file():
        return f"(file not found: {p})"

    desc = chart_description.strip() or "bar chart"
    prompt = (
        f"This image shows a page from a Treasury Bulletin PDF.\n"
        f"Please read the {desc}.\n"
        "For a bar chart with quarterly GDP data, list all labeled values:\n"
        "  Year: Q1=<value>, Q2=<value>, Q3=<value>, Q4=<value>\n"
        "Read the data labels directly off the bars. Be precise."
    )

    results: list[str] = []

    # Try embedded chart image first (often higher quality)
    if use_embedded:
        try:
            embedded = _extract_embedded_images(p, page_number)
            for i, img_bytes in enumerate(embedded):
                try:
                    text = _ask_vision(img_bytes, prompt)
                    results.append(f"[Embedded image {i+1}]\n{text}")
                except Exception as exc:
                    results.append(f"[Embedded image {i+1} error: {exc}]")
        except Exception as exc:
            results.append(f"[Embedded extraction error: {exc}]")

    # Also try rendering the full page
    try:
        page_png = _render_pdf_page(p, page_number)
        text = _ask_vision(page_png, prompt)
        results.append(f"[Full page render]\n{text}")
    except Exception as exc:
        results.append(f"[Page render error: {exc}]")

    return "\n\n---\n\n".join(results) if results else "(no chart data extracted)"
