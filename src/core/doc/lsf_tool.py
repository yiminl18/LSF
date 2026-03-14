"""
LSF visual pattern extraction tool.

Extracts visual features from PDFs using the LSF (Light-weight Structure Fusion)
method, including font size, bold, centering, and other visual attributes.

Main function:
- to_json(): Extract PDF visual patterns and save as JSON.
"""

import json
from pathlib import Path
from typing import Optional

from core.config import DEFAULT_OUTPUT_DIR
from core.doc.pdf_extraction import phrase_visual_pattern_extraction


def to_json(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """
    Extract visual pattern features from a PDF and save as JSON.

    Uses OCR and pdfplumber to extract text along with its visual attributes
    (font, size, position, etc.).

    Args:
        input_path: Path to the input PDF file.
        output_path: Path for the output file (auto-generated if None).
        output_dir: Output directory (default: 'result').
        verbose: Whether to print processing logs.

    Returns:
        Path to the output file.
    """
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    text = phrase_visual_pattern_extraction(str(input_file), verbose=verbose)
    json_content = json.dumps(text, ensure_ascii=False, indent=2)

    if output_path is None:
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        result_dir = Path(output_dir)
        output_file = result_dir / f"{input_file.stem}_lsf.json"
    else:
        output_file = Path(output_path).expanduser().resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json_content, encoding="utf-8")
    return output_file


if __name__ == "__main__":
    input_pdf = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
    json_path = to_json(input_pdf)
    print(f"JSON saved to: {json_path}")
