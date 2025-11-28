import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from core.docling_tool import to_json as docling_to_json
from core.lsf_tool import to_json as lsf_to_json
from core.text_summary import summarize_text


def normalize_text(text: str) -> str:
    """Normalize text: keep only letters for matching.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text (lowercase, letters only)
    """
    # Keep only letters (a-z, A-Z), remove all other characters (punctuation, spaces, numbers, etc.)
    normalized = re.sub(r'[^a-zA-Z]', '', text)
    # Convert to lowercase for matching
    return normalized.lower()


def split_docling_text(text: str) -> List[str]:
    """Split docling text by spaces, preserving docling's tokenization strategy.
    
    Args:
        text: Text to split
        
    Returns:
        List of words
    """
    # Simple space-based tokenization
    words = text.split()
    return [word for word in words if word.strip()]


def find_matching_lsf_words(
    docling_text: str,
    docling_bbox: Dict[str, float],
    docling_page: int,
    lsf_words: List[Dict[str, Any]],
    tolerance: float = 10.0
) -> Optional[List[Dict[str, Any]]]:
    """
    Find matching LSF word sequence.
    
    Args:
        docling_text: Complete text from Docling
        docling_bbox: Docling bbox {l, t, r, b}
        docling_page: Docling page number (1-indexed)
        lsf_words: List of all LSF words
        tolerance: Tolerance for left/right boundary matching (pixels)
    
    Returns:
        Matching LSF word sequence, or None if not found
    """
    # Convert page number: docling (1-indexed) -> lsf (0-indexed)
    lsf_page = docling_page - 1
    
    # Get docling's left and right boundaries
    docling_left = docling_bbox['l']
    docling_right = docling_bbox['r']
    
    # Normalize docling text (for matching)
    docling_normalized = normalize_text(docling_text)
    
    # Filter LSF words on the same page
    page_lsf_words = [w for w in lsf_words if w['page'] == lsf_page]
    
    if not page_lsf_words:
        return None
    
    # Pre-normalize all LSF word texts (optimization)
    for word in page_lsf_words:
        if 'normalized_phrase' not in word:
            word['normalized_phrase'] = normalize_text(word['phrase'])
    
    # Try to find matching word sequence
    best_match = None
    best_score = float('inf')
    
    # Iterate through all possible starting positions
    for start_idx in range(len(page_lsf_words)):
        # Try to find matching word sequence (don't pre-filter first word's left boundary)
        matched_words = []
        lsf_text_so_far = ''  # Accumulated normalized text
        
        for lsf_idx in range(start_idx, len(page_lsf_words)):
            lsf_word = page_lsf_words[lsf_idx]
            matched_words.append(lsf_word)
            
            # Use pre-normalized text (optimization)
            lsf_text_so_far += lsf_word['normalized_phrase']
            
            # Check if text matches
            if docling_normalized == lsf_text_so_far:
                # Exact match, calculate boundaries and check
                min_left = min([w['bbox'][0] for w in matched_words])
                max_right = max([w['bbox'][2] for w in matched_words])
                
                left_error = abs(min_left - docling_left)
                right_error = abs(max_right - docling_right)
                
                if left_error <= tolerance and right_error <= tolerance:
                    # Exact match and boundary match, return directly (optimization)
                    return matched_words
            elif lsf_text_so_far.startswith(docling_normalized):
                # LSF text starts with docling text (prefix match)
                min_left = min([w['bbox'][0] for w in matched_words])
                max_right = max([w['bbox'][2] for w in matched_words])
                
                left_error = abs(min_left - docling_left)
                right_error = abs(max_right - docling_right)
                
                if left_error <= tolerance and right_error <= tolerance:
                    score = left_error + right_error
                    
                    if score < best_score:
                        best_score = score
                        best_match = matched_words.copy()
            
            # If right boundary exceeds docling_right + tolerance, stop searching
            if lsf_word['bbox'][2] > docling_right + tolerance:
                break
    
    return best_match


def map_lsf_to_docling(
    docling_path: str,
    lsf_path: str,
    output_path: Optional[str] = None,
    tolerance: float = 10.0,
    summary_max_length: int = 100
) -> Path:
    """
    Map LSF visual information to Docling header texts.
    
    Args:
        docling_path: Path to Docling JSON file
        lsf_path: Path to LSF JSON file
        output_path: Output file path (if None, auto-generate)
        tolerance: Tolerance for left/right boundary matching (pixels)
        summary_max_length: Maximum length of summary, default 100
    
    Returns:
        Path to the output file
    """
    # Load files
    with open(docling_path, 'r', encoding='utf-8') as f:
        docling_data = json.load(f)
    
    with open(lsf_path, 'r', encoding='utf-8') as f:
        lsf_words = json.load(f)
    
    # Save original texts array for extracting text_span
    all_texts = docling_data.get('texts', [])
    
    # Extract all text items with label containing "header", and record their positions in original array
    header_texts = []
    header_indices = []  # Record index of each header in all_texts
    for idx, text_item in enumerate(all_texts):
        label = text_item.get('label', '')
        if 'header' in label.lower():
            header_texts.append(text_item)
            header_indices.append(idx)
    
    print(f"Found {len(header_texts)} header text items")
    
    # Add visual features to each header text
    for header_item in header_texts:
        text = header_item.get('text', '')
        if not text:
            continue
        
        # Split by spaces
        docling_words = split_docling_text(text)
        
        # Get bbox and page number
        prov = header_item.get('prov', [])
        if not prov:
            continue
        
        prov_item = prov[0]  # Take the first prov
        bbox = prov_item.get('bbox', {})
        page_no = prov_item.get('page_no', 1)
        
        # Find matching LSF word sequence
        matched_lsf_words = find_matching_lsf_words(
            text,
            bbox,
            page_no,
            lsf_words,
            tolerance
        )
        
        if matched_lsf_words:
            # If matching LSF word sequence found, use first LSF word's features for entire header
            # No need to tokenize or create words field
            first_lsf_word = matched_lsf_words[0]
            header_item['font'] = first_lsf_word.get('font')
            header_item['size'] = first_lsf_word.get('size')
            header_item['bold'] = first_lsf_word.get('bold')
            header_item['all_cap'] = first_lsf_word.get('all_cap')
            header_item['num_st'] = first_lsf_word.get('num_st')
            header_item['is_center'] = first_lsf_word.get('is_center')
            header_item['is_underline'] = first_lsf_word.get('is_underline')
            header_item['lsf_matched'] = True
        else:
            # If no match found, add match status indicator
            header_item['lsf_matched'] = "not matched"
    
    # Add text_span field to each header
    for header_idx, header_item in enumerate(header_texts):
        # Find current header's position in original array
        current_header_idx = header_indices[header_idx]
        
        # Find next header's position (if exists)
        if header_idx + 1 < len(header_indices):
            next_header_idx = header_indices[header_idx + 1]
        else:
            next_header_idx = len(all_texts)
        
        # Extract all items with label "text" between current header and next header
        text_span_parts = []
        for text_idx in range(current_header_idx + 1, next_header_idx):
            if text_idx < len(all_texts):
                text_item = all_texts[text_idx]
                label = text_item.get('label', '')
                if label == 'text':  # Only extract items with label "text"
                    text_content = text_item.get('text', '')
                    if text_content:
                        text_span_parts.append(text_content)
        
        # Join all text content
        text_span_content = ' '.join(text_span_parts)
        header_item['text_span'] = text_span_content
        
        # Generate summary
        if text_span_content:
            summary = summarize_text(text_span_content, max_length=summary_max_length)
            header_item['summary'] = summary if summary else None
        else:
            header_item['summary'] = None
    
    # Clean up any words fields that may exist in header_item
    for header_item in header_texts:
        if 'words' in header_item:
            del header_item['words']
    
    # Keep only header-related data, remove all other content
    # 1. Keep only header items in texts
    docling_data['texts'] = header_texts
    
    # 2. Update body.children to only include header references
    header_refs = []
    for header_item in header_texts:
        self_ref = header_item.get('self_ref', '')
        if self_ref:
            header_refs.append({'$ref': self_ref})
    docling_data['body']['children'] = header_refs
    
    # 3. Delete other unnecessary data structures
    if 'tables' in docling_data:
        docling_data['tables'] = []
    if 'groups' in docling_data:
        docling_data['groups'] = []
    if 'pictures' in docling_data:
        docling_data['pictures'] = []
    
    # 4. Clear furniture's children
    if 'furniture' in docling_data:
        docling_data['furniture']['children'] = []
    
    print(f"Filtered, keeping only {len(header_texts)} header items")
    
    # Save results
    if output_path is None:
        docling_file = Path(docling_path)
        output_path = docling_file.parent / f"{docling_file.stem}_merged.json"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(docling_data, f, ensure_ascii=False, indent=2)
    
    print(f"Merged result saved to: {output_path}")
    return output_path


def process_pdf_with_both_tools(
    input_pdf_path: str,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    tolerance: float = 10.0,
    summary_max_length: int = 100
) -> Path:
    """
    Complete processing pipeline: call docling_tool and lsf_tool, then map and merge results.
    If processed files already exist, return directly.
    
    Args:
        input_pdf_path: Path to input PDF file
        output_path: Output file path (if None, auto-generate to output_dir)
        output_dir: Directory for output files (default: 'result')
        tolerance: Tolerance for left/right boundary matching (pixels)
        summary_max_length: Maximum length of summary, default 100
    
    Returns:
        Path to the output file
    """
    input_file = Path(input_pdf_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    # Set output directory
    if output_dir is None:
        output_dir = "output/paper"
    result_dir = Path(output_dir)
    
    # Check if processed files already exist
    docling_json_path = result_dir / f"{input_file.stem}_docling.json"
    lsf_json_path = result_dir / f"{input_file.stem}_lsf.json"
    
    if output_path is None:
        merged_json_path = result_dir / f"{input_file.stem}_merged.json"
    else:
        merged_json_path = Path(output_path)
    
    # If all files exist, return directly
    if docling_json_path.exists() and lsf_json_path.exists() and merged_json_path.exists():
        print(f"Found existing processed files, returning: {merged_json_path}")
        return merged_json_path
    
    print(f"Starting PDF processing: {input_file}")
    
    # 1. Call docling_tool to generate JSON (if not exists)
    if not docling_json_path.exists():
        print("Calling docling_tool...")
        docling_json_path = docling_to_json(str(input_file), output_dir=output_dir)
        print(f"Docling JSON saved to: {docling_json_path}")
    else:
        print(f"Docling JSON already exists: {docling_json_path}")
    
    # 2. Call lsf_tool to generate JSON (if not exists)
    if not lsf_json_path.exists():
        print("Calling lsf_tool...")
        lsf_json_path = lsf_to_json(str(input_file), output_dir=output_dir)
        print(f"LSF JSON saved to: {lsf_json_path}")
    else:
        print(f"LSF JSON already exists: {lsf_json_path}")
    
    # 3. Map and merge (if merged_json doesn't exist)
    if not merged_json_path.exists():
        print("Mapping and merging...")
        merged_path = map_lsf_to_docling(
            str(docling_json_path),
            str(lsf_json_path),
            str(merged_json_path),
            tolerance,
            summary_max_length
        )
        print(f"Complete! Merged result saved to: {merged_path}")
    else:
        print(f"Merged JSON already exists: {merged_json_path}")
        merged_path = merged_json_path
    
    return merged_path


if __name__ == "__main__":
    # Example usage
    # Method 1: Use existing JSON files
    # docling_path = "result/NIPS-2017-attention-is-all-you-need-Paper.json"
    # lsf_path = "result/NIPS-2017-attention-is-all-you-need-Paper_lsf.json"
    
    # output_path = map_lsf_to_docling(docling_path, lsf_path)
    # print(f"Complete! Output file: {output_path}")
    
    # Method 2: Complete processing pipeline starting from PDF
    input_pdf = "pdfs/MICROSOFT_2018_10K.pdf"
    merged_path = process_pdf_with_both_tools(input_pdf)
    print(f"Complete! Merged result: {merged_path}")

