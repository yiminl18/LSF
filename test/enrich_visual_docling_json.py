import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.map_lsf_to_docling import process_pdf_with_both_tools


def enrich_pdf(input_pdf_path: str, output_dir: Optional[str] = None, output_path: Optional[str] = None) -> Path:
    """Enrich a PDF by processing it with both docling and LSF tools, then mapping visual features.
    
    Args:
        input_pdf_path: Path to the input PDF file
        output_dir: Directory for output files (default: 'result')
        output_path: Output file path (if None, auto-generate to output_dir)
        
    Returns:
        Path to the output merged JSON file
    """
    result_path = process_pdf_with_both_tools(
        input_pdf_path=input_pdf_path,
        output_path=output_path,
        output_dir=output_dir
    )
    return result_path


if __name__ == "__main__":
    # Pick the first paper PDF from data/paper
    paper_dir = Path("data/paper")
    pdf_files = sorted(list(paper_dir.glob("*.pdf")))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {paper_dir}")
    
    # Select the first PDF
    first_pdf = pdf_files[0]
    print(f"Selected first paper: {first_pdf}")
    
    # Process the file and save to result directory
    output_path = enrich_pdf(
        input_pdf_path=str(first_pdf),
        output_dir="output/paper"
    )
    print(f"Enrichment complete! Output saved to: {output_path}")

