"""Adapter: converts DocInputs to MDocAgent input format.

MDocAgent expects a tmp/<dataset>/ directory containing:
  - <doc_id>/pages/page_<N>.png  -- per-page images
  - <doc_id>/text.txt            -- full document text

Usage as standalone prep step:
    PYTHONPATH=src python -m agent.baselines.mdocagent.adapter \
        --config src/agent/config_pdfs_10doc.yaml \
        --query 0 --doc-id AMAZON_2015_10K
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agent.baselines.base import DocInputs

_DEFAULT_TMP_ROOT = Path(".cache/mdocagent/tmp")
_DEFAULT_DATASET_NAME = "pdfs"
_RENDER_DPI = 100


def prepare_inputs(
    doc_inputs: DocInputs,
    doc_id: str,
    dataset_name: str = _DEFAULT_DATASET_NAME,
    tmp_root: Path = _DEFAULT_TMP_ROOT,
) -> Path:
    """Render PDF pages and write text dump into MDocAgent's expected layout.

    Layout:
      tmp_root/<dataset_name>/<doc_id>/pages/page_<N>.png
      tmp_root/<dataset_name>/<doc_id>/text.txt

    Returns:
        doc_dir: Path to the <doc_id> directory.
    """
    doc_dir = tmp_root / dataset_name / doc_id
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Write text dump
    (doc_dir / "text.txt").write_text(doc_inputs.normalized_text, encoding="utf-8")

    # Render pages if PDF exists
    if doc_inputs.pdf_path.exists():
        try:
            import pypdfium2 as pdfium  # type: ignore[import]
            import io
            from PIL import Image  # type: ignore[import]

            doc = pdfium.PdfDocument(str(doc_inputs.pdf_path))
            scale = _RENDER_DPI / 72.0
            for page_no, page in enumerate(doc, start=1):
                bitmap = page.render(scale=scale)
                pil_image = bitmap.to_pil()
                out = pages_dir / f"page_{page_no:04d}.png"
                if not out.exists():
                    pil_image.save(out, format="PNG")
            doc.close()
        except Exception as exc:
            # Warn but continue; text-only mode still works
            print(f"[mdocagent/adapter] PDF render warning for {doc_id}: {exc}")

    return doc_dir


def main(argv: list[str] | None = None) -> None:
    """Standalone prep step: prepare MDocAgent inputs for one doc."""
    import yaml

    parser = argparse.ArgumentParser(description="Prepare MDocAgent inputs")
    parser.add_argument("--config", type=Path, default=Path("src/agent/config_pdfs_10doc.yaml"))
    parser.add_argument("--query", type=int, required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--tmp-root", type=Path, default=_DEFAULT_TMP_ROOT)
    args = parser.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from agent.baselines.loader import build_doc_inputs
    doc_inputs = build_doc_inputs(config, args.query, args.doc_id)
    doc_dir = prepare_inputs(
        doc_inputs,
        doc_id=args.doc_id,
        dataset_name=config.get("dataset", "pdfs"),
        tmp_root=args.tmp_root,
    )
    print(f"MDocAgent inputs prepared at: {doc_dir}")
    pages = list((doc_dir / "pages").glob("*.png"))
    print(f"  {len(pages)} page images, text.txt: {(doc_dir / 'text.txt').stat().st_size} bytes")


if __name__ == "__main__":
    main()
