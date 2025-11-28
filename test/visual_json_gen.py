import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.lsf_tool import to_json


def process_file(input_path: str, output_dir: Optional[str] = None) -> Path:
    """Process a file by converting it to JSON using lsf_tool.to_json.
    
    Args:
        input_path: Path to the input file
        output_dir: Directory to save output (default: 'test/paper')
        
    Returns:
        Path to the output JSON file
    """
    if output_dir is None:
        output_dir = "test/paper"
    
    output_path = to_json(input_path, output_dir=output_dir)
    return output_path


if __name__ == "__main__":
    # Pick the first paper PDF from data/paper
    paper_dir = Path("data/paper")
    pdf_files = sorted(list(paper_dir.glob("*.pdf")))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {paper_dir}")
    
    # Select the first PDF
    first_pdf = pdf_files[0]
    print(f"Selected first paper: {first_pdf}")
    
    # Process the file and save to test/paper
    output_path = process_file(str(first_pdf), output_dir="test/paper")
    print(f"JSON output saved to: {output_path}")

