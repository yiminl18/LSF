import json
from pathlib import Path
from typing import Optional

try:
    from docling.document_converter import DocumentConverter
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Docling is not installed or cannot be imported. Please run: pip install docling\nError: " + str(exc)
    )


def to_md(input_path: str, output_path: Optional[str] = None, output_dir: Optional[str] = None) -> Path:
    """Convert the specified file to Markdown and save to the specified path.
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the output file (if None, auto-generate)
        output_dir: Directory to save output (default: 'result')
        
    Returns:
        Path to the output file
    """
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    converter = DocumentConverter()
    conv_res = converter.convert(input_file)

    markdown_content = conv_res.document.export_to_markdown()
    
    if output_path is None:
        # Default: save to output_dir folder with same filename plus _docling suffix
        if output_dir is None:
            output_dir = "result"
        result_dir = Path(output_dir)
        output_file = result_dir / f"{input_file.stem}_docling.md"
    else:
        output_file = Path(output_path).expanduser().resolve()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(markdown_content, encoding="utf-8")
    return output_file


def to_json(input_path: str, output_path: Optional[str] = None, output_dir: Optional[str] = None) -> Path:
    """Convert the specified file to JSON and save to the specified path.
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the output file (if None, auto-generate)
        output_dir: Directory to save output (default: 'result')
        
    Returns:
        Path to the output file
    """
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    converter = DocumentConverter()
    conv_res = converter.convert(input_file)

    # Get document dictionary
    doc_dict = conv_res.document.export_to_dict()
    
    # Sort texts array by bbox from top to bottom
    if 'texts' in doc_dict:
        def get_sort_key(text_item):
            """Get sort key: first by page number, then by y coordinate (top to bottom)."""
            prov = text_item.get('prov', [])
            if not prov:
                return (0, 0)  # If no prov, put at the beginning
            
            prov_item = prov[0]  # Take the first prov
            page_no = prov_item.get('page_no', 0)
            bbox = prov_item.get('bbox', {})
            
            # BOTTOMLEFT coordinate system: larger t values mean higher on page
            # Use t value as y coordinate (descending sort, larger t first)
            t_value = bbox.get('t', 0)
            
            return (page_no, -t_value)  # Negative sign for descending sort
        
        doc_dict['texts'].sort(key=get_sort_key)
    
    json_content = json.dumps(doc_dict, ensure_ascii=False, indent=2)
    
    if output_path is None:
        # Default: save to output_dir folder with same filename plus _docling suffix
        if output_dir is None:
            output_dir = "result"
        result_dir = Path(output_dir)
        output_file = result_dir / f"{input_file.stem}_docling.json"
    else:
        output_file = Path(output_path).expanduser().resolve()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json_content, encoding="utf-8")
    return output_file


if __name__ == "__main__":
    # Example usage
    input_pdf = "statement_6418.pdf"

    # md_path = to_md(input_pdf)
    # print(f"Markdown saved to: {md_path}")
    
    json_path = to_json(input_pdf)
    print(f"JSON saved to: {json_path}")

