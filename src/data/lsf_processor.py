import json
from pathlib import Path
from typing import Optional

from core.tree_gen import phrase_visual_pattern_extraction


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

    # Use phrase_visual_pattern_extraction to extract text
    text = phrase_visual_pattern_extraction(str(input_file))

    json_content = json.dumps(text, ensure_ascii=False, indent=2)
    
    if output_path is None:
        # Default: save to output_dir folder with same filename plus _lsf suffix
        if output_dir is None:
            output_dir = "result"
        result_dir = Path(output_dir)
        output_file = result_dir / f"{input_file.stem}_lsf.json"
    else:
        output_file = Path(output_path).expanduser().resolve()
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json_content, encoding="utf-8")
    return output_file


if __name__ == "__main__":
    # Example usage
    input_pdf = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
    
    json_path = to_json(input_pdf)
    print(f"JSON saved to: {json_path}")

