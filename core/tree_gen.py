import json
from pathlib import Path
from typing import Dict, List, Any
import fitz
import pytesseract
from PIL import Image
import io
import os,re 
import pdfplumber
from typing import List, Dict, Any, Tuple

def is_valid_phrase(text: str) -> bool:
    """
    Check if a text phrase is valid (not empty, not just whitespace, not just special characters).
    
    Args:
        text (str): Text to validate
        
    Returns:
        bool: True if text is valid, False otherwise
    """
    if not text:
        return False
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check if empty after stripping
    if not text:
        return False
    
    # Check if only whitespace
    if text.isspace():
        return False
    
    # Check if only special/invisible characters (common in PDFs)
    # Remove common invisible characters and check if anything meaningful remains
    cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    cleaned_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', cleaned_text)  # Zero-width spaces
    
    if not cleaned_text.strip():
        return False
    
    # Check if the cleaned text has at least one printable character
    if not any(c.isprintable() for c in cleaned_text):
        return False
    
    return True 

def _is_bold_font(font_name: str) -> bool:
    """
    Check if font is bold based on font name.
    
    Args:
        font_name: Font name string
    
    Returns:
        True if font appears to be bold
    """
    font_lower = font_name.lower()
    bold_indicators = ['bold', 'b', 'heavy', 'black', 'demibold']
    return any(indicator in font_lower for indicator in bold_indicators)


def _is_all_caps(text: str) -> bool:
    """
    Check if text is all uppercase.
    
    Args:
        text: Text string
    
    Returns:
        True if text is all uppercase
    """
    if not text:
        return False
    return text.isupper() and text.isalpha()


def _starts_with_number(text: str) -> bool:
    """
    Check if text starts with a number.
    
    Args:
        text: Text string
    
    Returns:
        True if text starts with a number
    """
    if not text:
        return False
    return text[0].isdigit()


def _is_centered(bbox: Tuple[float, float, float, float], page_width: float, tolerance: float = 0.1) -> bool:
    """
    Check if phrase is centered on the page.
    
    Args:
        bbox: Bounding box coordinates
        page_width: Width of the page
        tolerance: Tolerance for centering (fraction of page width)
    
    Returns:
        True if phrase is centered
    """
    phrase_center = (bbox[0] + bbox[2]) / 2
    page_center = page_width / 2
    center_tolerance = page_width * tolerance
    
    return abs(phrase_center - page_center) <= center_tolerance


def _has_underline_word(word: Dict) -> bool:
    """
    Check if a word has underline.
    Note: pdfplumber doesn't always detect underlines reliably.
    
    Args:
        word: Word dictionary from pdfplumber
    
    Returns:
        True if underline is detected
    """
    # pdfplumber doesn't always provide underline information
    # This is a placeholder - in practice, you might need to analyze
    # the raw PDF objects or use additional heuristics
    return False


def _has_underline(words: List[Dict]) -> bool:
    """
    Check if any word in the phrase has underline.
    Note: pdfplumber doesn't always detect underlines reliably.
    
    Args:
        words: List of word dictionaries
    
    Returns:
        True if underline is detected
    """
    # pdfplumber doesn't always provide underline information
    # This is a placeholder - in practice, you might need to analyze
    # the raw PDF objects or use additional heuristics
    return False

def extract_phrases_with_pdfplumber(pdf_path: str, output_file: str = None) -> List[Dict[str, Any]]:
    """
    Extract words from a PDF using pdfplumber and save to local file.
    Based on the get_patterns approach - extracts individual words with their patterns.
    
    Args:
        pdf_path: Path to the PDF file
        output_file: Path to save the extracted words (optional)
    
    Returns:
        List of dictionaries containing word information with visual patterns
    """
    if output_file is None:
        # Generate output filename based on input PDF name
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = f"out/{base_name}_pdfplumber_words.json"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    words_data = []
    word_id = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Processing PDF with {len(pdf.pages)} pages...")
            
            for page_num, page in enumerate(pdf.pages):
                print(f"Processing page {page_num + 1}...")
                
                # Extract words with text flow and extra attributes
                words = page.extract_words(
                    use_text_flow=True, 
                    extra_attrs=["fontname", "size"]
                )
                
                for word in words:
                    # Skip empty words
                    if not word['text'].strip():
                        continue
                    
                    # Create bounding box
                    bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
                    
                    # Get font information
                    font_name = word.get('fontname', 'Unknown')
                    font_size = round(word.get('size', 12), 3)
                    
                    # Calculate visual pattern features
                    is_bold = _is_bold_font(font_name)
                    is_all_cap = _is_all_caps(word['text'])
                    starts_with_number = _starts_with_number(word['text'])
                    is_center = _is_centered(bbox, page.width)
                    is_underline = _has_underline_word(word)
                    
                    # Create word dictionary
                    word_dict = {
                        'id': word_id,
                        'phrase': word['text'],  # Keep 'phrase' key for compatibility
                        'bbox': bbox,
                        'page': page_num,
                        'font': font_name,
                        'size': font_size,
                        'bold': 1 if is_bold else 0,
                        'all_cap': 1 if is_all_cap else 0,
                        'num_st': 1 if starts_with_number else 0,
                        'is_center': 1 if is_center else 0,
                        'is_underline': 1 if is_underline else 0
                    }
                    
                    words_data.append(word_dict)
                    word_id += 1
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []
    
    return words_data

def phrase_visual_pattern_extraction(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract phrases and their visual patterns from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        List[Dict[str, Any]]: List of phrase objects with visual properties
        Each object contains:
        - phrase: extracted text
        - bbox: bounding box (x0, y0, x1, y1)
        - page: page number (0-indexed)
        - font: font type
        - size: font size
        - bold: 0 or 1 indicating if text is bold
        - is_underline: 0 or 1 indicating if text is underlined
        - all_cap: 0 or 1 indicating if all letters are capitalized
        - num_st: 0 or 1 indicating if phrase starts with a number
        - is_center: 0 or 1 indicating if phrase is centered on the page
    """
    
    def is_scanned_pdf(doc) -> bool:
        """Determine if PDF is scanned (image-based) or text-based."""
        first_page = doc[0]
        text = first_page.get_text()
        # If very little text is extracted, likely scanned
        return len(text.strip()) < 100
    
    def extract_text_from_scanned_pdf(doc) -> List[Dict[str, Any]]:
        """Extract text from scanned PDF using OCR."""
        phrases = []
        phrase_id = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get page dimensions
            rect = page.rect
            width, height = rect.width, rect.height
            
            # Convert page to image
            mat = fitz.Matrix(2, 2)  # Scale factor for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Use OCR to extract text with bounding boxes
            try:
                # Use pytesseract with detailed output
                ocr_data = pytesseract.image_to_data(
                    img, 
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'  # Assume uniform block of text
                )
                
                # Process OCR results
                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    # Filter out empty phrases, whitespace-only phrases, and special characters
                    if (is_valid_phrase(text) and int(ocr_data['conf'][i]) > 30):  # Confidence threshold
                        # Get bounding box (normalize to page coordinates)
                        x0 = ocr_data['left'][i] / 2  # Divide by scale factor
                        y0 = ocr_data['top'][i] / 2
                        x1 = x0 + (ocr_data['width'][i] / 2)
                        y1 = y0 + (ocr_data['height'][i] / 2)
                        
                        # Estimate font properties from OCR
                        font_size = ocr_data['height'][i] / 2
                        
                        # Simple heuristic for bold detection
                        # Higher confidence and larger height might indicate bold
                        is_bold = 1 if (int(ocr_data['conf'][i]) > 70 and font_size > 20) else 0
                        
                        # Simple heuristic for underline detection
                        # For OCR, we can't reliably detect underline, so set to 0
                        # In practice, underlined text might be detected as separate elements
                        is_underline = 0
                        
                        # Check if all letters are capitalized
                        all_cap = 1 if text.isupper() and text.isalpha() else 0
                        
                        # Check if phrase starts with a number
                        num_st = 1 if text and text[0].isdigit() else 0
                        
                        # Check if phrase is centered on the page
                        page_width = width
                        phrase_center_x = (x0 + x1) / 2
                        page_center_x = page_width / 2
                        # Heuristic: consider centered if within 20% of page center
                        center_threshold = page_width * 0.2
                        is_center = 1 if abs(phrase_center_x - page_center_x) <= center_threshold else 0
                        
                        phrases.append({
                            'id': phrase_id,
                            'phrase': text,
                            'bbox': (x0, y0, x1, y1),
                            'page': page_num,
                            'font': 'OCR_Detected',
                            'size': font_size,
                            'bold': is_bold,
                            'is_underline': is_underline,
                            'all_cap': all_cap,
                            'num_st': num_st,
                            'is_center': is_center
                        })
                        phrase_id += 1
                        
            except Exception as e:
                print(f"OCR failed for page {page_num}: {e}")
                continue
                
        return phrases
    

    try:
        # Open the PDF document
        doc = fitz.open(file_path)
        
        if not doc:
            raise ValueError("Could not open PDF file")
        
        # Determine if PDF is scanned or normal
        scanned = is_scanned_pdf(doc)
        
        if scanned:
            print("Detected scanned PDF - using OCR")
            phrases = extract_text_from_scanned_pdf(doc)
        else:
            print("Detected normal PDF - using pdfplumber")
            # Use pdfplumber for normal PDFs
            phrases = extract_phrases_with_pdfplumber(file_path)
        
        # Clean up
        doc.close()
        
        return phrases
        
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return []


def save_phrases_to_json(phrases: List[Dict[str, Any]], output_file: str):
    """Save extracted phrases to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(phrases, f, indent=2, ensure_ascii=False)

def calculate_bbox_overlap(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate the overlap area between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)
        
    Returns:
        Overlap area as a float
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    
    # Calculate intersection
    x0_intersect = max(x0_1, x0_2)
    y0_intersect = max(y0_1, y0_2)
    x1_intersect = min(x1_1, x1_2)
    y1_intersect = min(y1_1, y1_2)
    
    # Check if there's an intersection
    if x0_intersect >= x1_intersect or y0_intersect >= y1_intersect:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x1_intersect - x0_intersect) * (y1_intersect - y0_intersect)
    
    # Calculate union area
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - intersection_area
    
    # Return intersection over union (IoU)
    return intersection_area / union_area if union_area > 0 else 0.0

def find_matching_phrase(section_text: str, section_bbox: Tuple[float, float, float, float], 
                        section_page: int, phrases: List[Dict[str, Any]], 
                        overlap_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Find a phrase in the extracted phrases that matches the section header.
    
    Args:
        section_text: Text of the section header
        section_bbox: Bounding box of the section header
        section_page: Page number of the section header (0-indexed, matching phrases)
        phrases: List of extracted phrases with visual patterns
        overlap_threshold: Minimum overlap ratio to consider a match
        
    Returns:
        Matching phrase dictionary or None if no match found
    """
    best_match = None
    best_overlap = 0.0

    # if 'ABSTRACT' in section_text:
    #     print('section_text')
    #     print(section_text)
    #     print(section_bbox)
    #     print(section_page)
    
    # print('phrase_text')

    for phrase in phrases:
        
        # Check if page numbers match
        if phrase.get('page') != section_page:
            continue
        
        # Check if text matches (case-insensitive)
        phrase_text = phrase.get('phrase', '').strip()

        # Try different matching strategies
        section_text_lower = section_text.lower()
        phrase_text_lower = phrase_text.lower()
        
        # Strategy 1: Exact match
        if phrase_text_lower == section_text_lower:
            matched = True
        # Strategy 2: Section text starts with phrase (for multi-word section headers)
        elif section_text_lower.startswith(phrase_text_lower):
            matched = True
        # Strategy 3: Phrase text starts with section text (for concatenated phrases)
        elif phrase_text_lower.startswith(section_text_lower):
            matched = True
        # Strategy 4: Partial match for key words (for section headers like "1 INTRODUCTION")
        elif len(phrase_text.split()) >= 2 and any(word in section_text_lower for word in phrase_text_lower.split()):
            matched = True
        # Strategy 5: Remove whitespace and compare (for concatenated phrases)
        elif phrase_text_lower == section_text_lower.replace(' ', ''):
            matched = True
        else:
            matched = False
            
        if not matched:
            continue
        
        # Calculate bounding box overlap
        if 'abstract' in phrase_text:
            print(phrase_text)
            print(phrase.get('bbox'))
            print(section_bbox)

        phrase_bbox = phrase.get('bbox')
        if phrase_bbox:
            overlap = calculate_bbox_overlap(section_bbox, phrase_bbox)
            
            if overlap > best_overlap and overlap >= overlap_threshold:
                best_overlap = overlap
                best_match = phrase
    
    return best_match

def phrase_merge(phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive phrases with the same visual properties.
    
    Args:
        phrases (List[Dict[str, Any]]): List of phrase objects with visual properties
        
    Returns:
        List[Dict[str, Any]]: List of merged phrase objects
    """
    if not phrases:
        return []
    
    merged_phrases = []
    current_group = [phrases[0]]
    merged_id = 0
    
    for i in range(1, len(phrases)):
        current_phrase = phrases[i]
        last_phrase = current_group[-1]
        
        # Check if current phrase can be merged with the last phrase in current group
        can_merge = (
            current_phrase['page'] == last_phrase['page'] and
            current_phrase['font'] == last_phrase['font'] and
            current_phrase['size'] == last_phrase['size'] and
            current_phrase['bold'] == last_phrase['bold'] and
            current_phrase['is_underline'] == last_phrase['is_underline'] 
        )
        # and current_phrase['all_cap'] == last_phrase['all_cap']
        # Additional check: don't merge underlined and bold phrases if they're not in the same row
        if can_merge and (current_phrase['is_underline'] == 1 and current_phrase['bold'] == 1):
            # Check if phrases are in the same row (similar y-coordinates)
            current_y_center = (current_phrase['bbox'][1] + current_phrase['bbox'][3]) / 2
            last_y_center = (last_phrase['bbox'][1] + last_phrase['bbox'][3]) / 2
            
            # Use font size as tolerance for row detection
            font_size = current_phrase['size']
            y_tolerance = font_size * 0.5  # Allow half a font size difference
            
            # If y-coordinates differ significantly, don't merge
            if abs(current_y_center - last_y_center) > y_tolerance:
                can_merge = False
        
        # Additional check: don't merge bold phrases if they're not in the same row and first phrase is not close to right page boundary
        if can_merge and (current_phrase['bold'] == 1 and last_phrase['bold'] == 1):
            # Check if phrases are in the same row (similar y-coordinates)
            current_y_center = (current_phrase['bbox'][1] + current_phrase['bbox'][3]) / 2
            last_y_center = (last_phrase['bbox'][1] + last_phrase['bbox'][3]) / 2
            
            # Use font size as tolerance for row detection
            font_size = current_phrase['size']
            y_tolerance = font_size * 0.5  # Allow half a font size difference
            
            # If y-coordinates differ significantly (not in same row)
            if abs(current_y_center - last_y_center) > y_tolerance:
                # Check if the first phrase (last_phrase) is close to the right page boundary
                # We need to estimate page width - use a reasonable default or get from context
                # For now, assume page width is around 612 points (standard US Letter)
                page_width = 612  # This could be made more dynamic
                
                # Calculate distance from right edge of first phrase to right page boundary
                first_phrase_right_edge = last_phrase['bbox'][2]
                distance_to_right_boundary = page_width - first_phrase_right_edge
                
                # If first phrase is not close to right boundary (more than 50 points away), don't merge
                if distance_to_right_boundary > 50:
                    can_merge = False
        
        if can_merge:
            # Add to current group for merging
            current_group.append(current_phrase)
        else:
            # Merge current group and add to result
            merged_phrase = merge_phrase_group(current_group, merged_id)
            merged_phrases.append(merged_phrase)
            merged_id += 1
            
            # Start new group with current phrase
            current_group = [current_phrase]
    
    # Don't forget to merge the last group
    if current_group:
        merged_phrase = merge_phrase_group(current_group, merged_id)
        merged_phrases.append(merged_phrase)
    
    return merged_phrases

def merge_phrase_group(phrase_group: List[Dict[str, Any]], merged_id: int) -> Dict[str, Any]:
    """
    Merge a group of phrases with the same visual properties into a single phrase.
    
    Args:
        phrase_group (List[Dict[str, Any]]): List of phrases to merge
        merged_id (int): ID to assign to the merged phrase
        
    Returns:
        Dict[str, Any]: Merged phrase object
    """
    if len(phrase_group) == 1:
        # Update the ID for single phrases
        result = phrase_group[0].copy()
        result['id'] = merged_id
        return result
    
    # Merge text
    merged_text = " ".join([phrase['phrase'] for phrase in phrase_group])
    
    # Merge bounding boxes
    x0 = min(phrase['bbox'][0] for phrase in phrase_group)
    y0 = min(phrase['bbox'][1] for phrase in phrase_group)
    x1 = max(phrase['bbox'][2] for phrase in phrase_group)
    y1 = max(phrase['bbox'][3] for phrase in phrase_group)
    merged_bbox = (x0, y0, x1, y1)
    
    # Use properties from the first phrase (they should all be the same)
    first_phrase = phrase_group[0]
    
    # Determine if merged phrase is centered
    # Calculate center of merged bbox
    merged_center_x = (x0 + x1) / 2
    # We need page width to determine centering - use from first phrase
    # This is approximate since we don't have page width here
    # For now, use the original is_center value from first phrase
    merged_is_center = first_phrase['is_center']
    
    # Determine if merged phrase starts with number
    merged_num_st = 1 if merged_text and merged_text[0].isdigit() else 0
    
    # Determine if merged phrase is all caps
    merged_all_cap = 1 if merged_text.isupper() and merged_text.isalpha() else 0
    
    return {
        'id': merged_id,
        'phrase': merged_text,
        'bbox': merged_bbox,
        'page': first_phrase['page'],
        'font': first_phrase['font'],
        'size': first_phrase['size'],
        'bold': first_phrase['bold'],
        'is_underline': first_phrase['is_underline'],
        'all_cap': merged_all_cap,
        'num_st': merged_num_st,
        'is_center': merged_is_center
    }

def enrich_visual_cues(tree: Dict[str, Any], phrases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enrich section header nodes with visual pattern information from extracted phrases.
    
    Args:
        tree: Tree structure returned by process_json_file
        phrases: List of phrases returned by phrase_visual_pattern_extraction
        
    Returns:
        Modified tree with visual pattern information added to section headers
    """
    if 'texts' not in tree:
        print("Warning: No 'texts' array found in the tree")
        return tree
    
    texts = tree['texts']
    enriched_count = 0
    
    for i, node in enumerate(texts):
        if node.get('label') == 'section_header':
            section_text = node.get('text', '')

            
            # Extract bounding box and page from the node's prov field
            bbox = None
            page_num = None
            
            if 'prov' in node and node['prov']:
                prov = node['prov'][0]  # Take the first provision
                bbox_data = prov.get('bbox', {})
                if bbox_data:
                    # Convert tree bbox format (dict with l,t,r,b) to list format [x0,y0,x1,y1]
                    # Tree uses BOTTOMLEFT origin: t=top (distance from bottom), b=bottom (distance from bottom)
                    # Phrases use TOPLEFT origin: y0=top (distance from top), y1=bottom (distance from top)
                    # Need to convert tree coordinates to phrase coordinates
                    
                    # Get page height to convert coordinates
                    page_height = 792.0  # Standard PDF page height, adjust if needed
                    
                    tree_left = bbox_data.get('l', 0)
                    tree_top = bbox_data.get('t', 0)      # Distance from bottom
                    tree_right = bbox_data.get('r', 0)
                    tree_bottom = bbox_data.get('b', 0)   # Distance from bottom
                    
                    # Convert to TOPLEFT origin (like phrases)
                    phrase_x0 = tree_left
                    phrase_y0 = page_height - tree_top      # Convert from bottom to top
                    phrase_x1 = tree_right
                    phrase_y1 = page_height - tree_bottom   # Convert from bottom to top
                    
                    bbox = [phrase_x0, phrase_y0, phrase_x1, phrase_y1]
                # Convert page_no from 1-indexed (tree) to 0-indexed (phrases)
                page_num = prov.get('page_no', 1) - 1
            
            if bbox and page_num is not None:
                # Find matching phrase
                matching_phrase = find_matching_phrase(section_text, bbox, page_num, phrases)
                
                if matching_phrase:
                    # Add visual pattern information to the node
                    visual_pattern = {
                        'font': matching_phrase.get('font', 'Unknown'),
                        'size': matching_phrase.get('size', 0),
                        'bold': matching_phrase.get('bold', 0),
                        'all_cap': matching_phrase.get('all_cap', 0),
                        'num_st': matching_phrase.get('num_st', 0),
                        'is_center': matching_phrase.get('is_center', 0),
                        'is_underline': matching_phrase.get('is_underline', 0)
                    }
                    
                    texts[i]['visual_pattern'] = visual_pattern
                    enriched_count += 1
                    print(f"Enriched section header: '{section_text[:50]}...' with visual pattern")
                else:
                    print(f"No matching phrase found for section header: '{section_text[:50]}...'")
            else:
                print(f"Could not extract bbox/page for section header: '{section_text[:50]}...'")
    
    print(f"Enriched {enriched_count} section headers with visual patterns")
    return tree

def add_flat_edge(tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a JSON tree structure to add hierarchical relationships based on section_header nodes.
    Now processes ALL node types: texts, tables, pictures, groups, key_value_items, form_items.
    
    Args:
        tree: Dictionary containing the JSON structure with various node arrays
        
    Returns:
        Modified tree with updated parent-child relationships for all node types
    """
    # Define all node array types to process
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    
    # First, collect all nodes from all arrays and sort them by page and position
    all_nodes = []
    
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                # Add array type information to each node for later processing
                node['_array_type'] = array_name
                all_nodes.append(node)
    
    if not all_nodes:
        print("Warning: No nodes found in any of the arrays")
        return tree
    
    # Sort all nodes by page number and bounding box coordinates
    all_nodes = sort_nodes_by_page_and_bbox(all_nodes)
    
    # Track the current section header node and its children
    current_section_header = None
    current_section_children = []
    
    # Process each node to establish parent-child relationships
    for node in all_nodes:
        node_label = node.get('label', '')
        
        if node_label == 'section_header':
            # If we have a previous section header, update its children array
            if current_section_header is not None:
                # Find and update the previous section header's children
                array_type = current_section_header.get('_array_type', 'texts')
                if array_type in tree:
                    for j, existing_node in enumerate(tree[array_type]):
                        if existing_node['self_ref'] == current_section_header['self_ref']:
                            tree[array_type][j]['children'] = current_section_children
                            break
            
            # Found a new section header, update current section
            current_section_header = node
            current_section_children = []
            # Section headers don't get parents
            
        elif current_section_header is not None:
            # This is a non-section_header node, add it as child of current section
            node['parent'] = {'$ref': current_section_header['self_ref']}
            
            # Add this node to the current section's children
            current_section_children.append({'$ref': node['self_ref']})
    
    # Don't forget to update the last section header's children
    if current_section_header is not None:
        array_type = current_section_header.get('_array_type', 'texts')
        if array_type in tree:
            for j, existing_node in enumerate(tree[array_type]):
                if existing_node['self_ref'] == current_section_header['self_ref']:
                    tree[array_type][j]['children'] = current_section_children
                    break
    
    # Remove the temporary _array_type field from all nodes
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                if '_array_type' in node:
                    del node['_array_type']
    
    return tree

def process_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and process a JSON file using add_flat_edge function.
    Now processes ALL node types: texts, tables, pictures, groups, key_value_items, form_items.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Processed tree structure with all node types sorted and processed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = json.load(f)
        
        # Sort all node types based on page number and bounding box
        node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
        for array_name in node_arrays:
            if array_name in tree and tree[array_name]:
                tree[array_name] = sort_nodes_by_page_and_bbox(tree[array_name])
        
        processed_tree = add_flat_edge(tree)
        return processed_tree
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {}

def sort_nodes_by_page_and_bbox(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort nodes based on page number and bounding box coordinates.
    First sort by page number, then by y-coordinate (t), then by x-coordinate (l).
    
    Args:
        nodes: List of node dictionaries
        
    Returns:
        Sorted list of nodes
    """
    def get_sort_key(node):
        # Get page number from prov field
        page_num = 0  # default page number
        y_coord = 0   # default y coordinate
        x_coord = 0   # default x coordinate
        
        if 'prov' in node and node['prov']:
            prov = node['prov'][0]  # Take the first provision
            page_num = prov.get('page_no', 0)
            
            # Get coordinates from bounding box
            bbox_data = prov.get('bbox', {})
            if bbox_data:
                # Use top coordinate (t) as y-coordinate for sorting
                # Note: t is distance from bottom, so we need to convert to top-based
                y_coord = bbox_data.get('t', 0)
                # Use left coordinate (l) as x-coordinate for sorting
                x_coord = bbox_data.get('l', 0)
        
        return (page_num, -y_coord, x_coord)
    
    # Sort nodes using the sort key
    sorted_nodes = sorted(nodes, key=get_sort_key)
    return sorted_nodes

def save_processed_tree(tree: Dict[str, Any], output_path: str) -> bool:
    """
    Save the processed tree to a JSON file.
    
    Args:
        tree: Processed tree structure
        output_path: Path where to save the output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")
        return False

def get_node_text(tree: Dict[str, Any], node_id: str) -> str:
    """
    Get the text content of a specific node by its ID.
    Now searches ALL node types: texts, tables, pictures, groups, key_value_items, form_items.
    Handles different node types appropriately (tables, images, etc.).
    
    Args:
        tree: Tree structure containing the nodes
        node_id: The ID of the node (e.g., "#/texts/0", "#/tables/0", etc.)
        
    Returns:
        The text content of the node, or appropriate placeholder for non-text nodes
    """
    # Define all node array types to search
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    
    # Search for the node with the given ID in all arrays
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                if node.get('self_ref') == node_id:
                    # Handle different node types appropriately
                    if array_name == 'texts':
                        return node.get('text', '')
                    elif array_name == 'tables':
                        # For tables, use the new plain text table function
                        return get_plain_text_table(tree, node_id)
                    elif array_name == 'pictures':
                        # For images, return a placeholder
                        return '[IMAGE]'
                    elif array_name == 'groups':
                        # For groups, try to get text or return placeholder
                        if 'text' in node:
                            return node.get('text', '')
                        else:
                            return '[GROUP]'
                    elif array_name in ['key_value_items', 'form_items']:
                        # For key-value pairs and form items, try to extract meaningful text
                        if 'text' in node:
                            return node.get('text', '')
                        elif 'key' in node and 'value' in node:
                            return f"{node.get('key', '')}: {node.get('value', '')}"
                        else:
                            return '[FORM_ITEM]'
                    else:
                        # Fallback for any other node type
                        return node.get('text', '')
    
    # Node not found
    print(f"Warning: Node with ID '{node_id}' not found in any node arrays")
    return ""

def get_node_text_only(tree: Dict[str, Any], node_id: str) -> str:
    """
    Very simple helper to get only the node's own text (no children aggregation).
    Returns an empty string if the node or its text field is not found.
    """
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                if node.get('self_ref') == node_id:
                    return node.get('text', '') or ''
    return ''

def get_node_page_info(tree: Dict[str, Any], node_id: str) -> Tuple[int, int]:
    """
    Return (current_node_page_no, total_pages) where total_pages is the max page_no across all nodes.
    If node not found or page_no missing, current_node_page_no is -1. If no pages found at all, total_pages is 0.
    """
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    current_page = -1
    max_page = 0

    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                # Track max page across all nodes
                if 'prov' in node and node['prov']:
                    prov = node['prov'][0]
                    page_no = prov.get('page_no')
                    if isinstance(page_no, int):
                        if page_no > max_page:
                            max_page = page_no

                # Check if this is the target node
                if current_page == -1 and node.get('self_ref') == node_id:
                    if 'prov' in node and node['prov']:
                        page_no = node['prov'][0].get('page_no')
                        if isinstance(page_no, int):
                            current_page = page_no

    return current_page, max_page

def get_node_info(tree: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific node by its ID.
    Now searches ALL node types: texts, tables, pictures, groups, key_value_items, form_items.
    
    Args:
        tree: Tree structure containing the nodes
        node_id: The ID of the node (e.g., "#/texts/0", "#/tables/0", etc.)
        
    Returns:
        Dictionary containing node information, or empty dict if node not found
    """
    # Define all node array types to search
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    
    # Search for the node with the given ID in all arrays
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                if node.get('self_ref') == node_id:
                    # Return the complete node information
                    return node.copy()
    
    # Node not found
    print(f"Warning: Node with ID '{node_id}' not found in any node arrays")
    return {}

def print_all_node_ids(processed_json_path: str) -> None:
    """
    Print all node IDs sequentially from a processed JSON file.
    Now includes ALL node types: texts, tables, pictures, groups, key_value_items, form_items.
    
    Args:
        processed_json_path: Path to the processed JSON file
    """
    try:
        # Load the processed JSON file
        with open(processed_json_path, 'r', encoding='utf-8') as f:
            tree = json.load(f)
        
        # Define all node array types to process
        node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
        
        # Collect all nodes from all arrays
        all_nodes = []
        for array_name in node_arrays:
            if array_name in tree and tree[array_name]:
                for node in tree[array_name]:
                    all_nodes.append(node)
        
        if not all_nodes:
            print("Warning: No nodes found in any arrays in the processed JSON file")
            return
        
        # Sort all nodes by page and position
        all_nodes = sort_nodes_by_page_and_bbox(all_nodes)
        
        print(f"Total nodes found: {len(all_nodes)}")
        print("=" * 50)
        
        # Print all node IDs sequentially
        for i, node in enumerate(all_nodes):
            node_id = node.get('self_ref', f'unknown_{i}')
            print(f"{i+1:3d}. {node_id}")
        
        print("=" * 50)
        print(f"Printed {len(all_nodes)} node IDs")
        
    except FileNotFoundError:
        print(f"Error: File {processed_json_path} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {processed_json_path}: {e}")
    except Exception as e:
        print(f"Error reading file {processed_json_path}: {e}")

def print_all_node_ids_from_base_path(base_json_path: str) -> None:
    """
    Print all node IDs from a processed JSON file, automatically constructing the processed file path.
    
    Args:
        base_json_path: Path to the base JSON file (e.g., "file.json")
    """
    # Construct the processed JSON path
    processed_path = Path(base_json_path).with_name(Path(base_json_path).stem + '_processed.json')
    print(f"Reading from: {processed_path}")
    print_all_node_ids(str(processed_path))

def print_all_node_ids_simple(tree: Dict[str, Any]) -> None:
    """
    Print only node IDs sequentially from a tree structure.
    Now includes ALL node types: texts, tables, pictures, groups, key_value_items, form_items.
    
    Args:
        tree: Tree structure containing the nodes
    """
    # Define all node array types to process
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    
    # Collect all nodes from all arrays
    all_nodes = []
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                all_nodes.append(node)
    
    if not all_nodes:
        return
    
    # Sort all nodes by page and position
    all_nodes = sort_nodes_by_page_and_bbox(all_nodes)
    
    # Print only node IDs, one per line
    for node in all_nodes:
        node_id = node.get('self_ref', '')
        if node_id:
            print(node_id)

def print_tree_structure(tree: Dict[str, Any], max_depth: int = 3) -> None:
    """
    Print a hierarchical view of the tree structure showing section headers and their child IDs.
    Now includes ALL node types: texts, tables, pictures, groups, key_value_items, form_items.
    
    Args:
        tree: Tree structure to print
        max_depth: Maximum depth to print (to avoid overwhelming output)
    """
    # Define all node array types to process
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    
    # Collect all nodes from all arrays
    all_nodes = []
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                all_nodes.append(node)
    
    if not all_nodes:
        print("No nodes found in tree")
        return
    
    section_headers = []
    
    # Find all section headers from all node types
    for node in all_nodes:
        if node.get('label') == 'section_header':
            section_headers.append({
                'id': node['self_ref'],
                'text': node.get('text', ''),
                'children': node.get('children', [])
            })
    
    print(f"Found {len(section_headers)} section headers:")
    print("=" * 80)
    
    for i, section in enumerate(section_headers):
        # Get page number for this section
        page_no = "?"
        for node in all_nodes:
            if node['self_ref'] == section['id']:
                if 'prov' in node and node['prov']:
                    prov = node['prov'][0]
                    page_no = str(prov.get('page_no', '?'))
                break
        
        print(f"\n{i+1}. Section Header: {section['id']} [Page: {page_no}]")
        print(f"   Text: {section['text']}")
        print(f"   Children count: {len(section['children'])}")
        
        if section['children']:
            print("   Child IDs:")
            for j, child_ref in enumerate(section['children']):
                child_id = child_ref.get('$ref', 'unknown')
                # Find the actual child node to get its label and text
                child_node = None
                for node in all_nodes:
                    if node['self_ref'] == child_id:
                        child_node = node
                        break
                
                if child_node:
                    child_label = child_node.get('label', 'unknown')
                    child_text = child_node.get('text', '')[:60]
                    if len(child_node.get('text', '')) > 60:
                        child_text += "..."
                    
                    # Get page number for child node
                    child_page_no = "?"
                    if 'prov' in child_node and child_node['prov']:
                        prov = child_node['prov'][0]
                        child_page_no = str(prov.get('page_no', '?'))
                    
                    print(f"     {j+1}. {child_id} ({child_label}) [Page: {child_page_no}]: {child_text}")
                else:
                    print(f"     {j+1}. {child_id} (not found in any node arrays)")
        else:
            print("   No children")
        
        print("-" * 60)
    
    # Also show nodes without section headers (orphans)
    orphan_nodes = [node for node in all_nodes if node.get('label') != 'section_header' and 'parent' not in node]
    if orphan_nodes:
        print(f"\nOrphan nodes (no section header parent): {len(orphan_nodes)}")
        for i, orphan in enumerate(orphan_nodes[:10]):  # Limit to first 10
            orphan_text = orphan.get('text', '')[:50]
            if len(orphan.get('text', '')) > 50:
                orphan_text += "..."
            
            # Get page number for orphan node
            orphan_page_no = "?"
            if 'prov' in orphan and orphan['prov']:
                prov = orphan['prov'][0]
                orphan_page_no = str(prov.get('page_no', '?'))
            
            print(f"  {i+1}. {orphan['self_ref']} ({orphan.get('label', 'unknown')}) [Page: {orphan_page_no}]: {orphan_text}")
        
        if len(orphan_nodes) > 10:
            print(f"  ... and {len(orphan_nodes) - 10} more orphan nodes")

def get_section_headers_by_id_order(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all nodes with label 'section_header' and return them in order of their IDs.
    
    Args:
        tree: Tree structure containing the nodes
        
    Returns:
        List of section header nodes sorted by their ID order
    """
    # Define all node array types to search
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    
    section_headers = []
    
    # Search for section headers in all node arrays
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                if node.get('label') == 'section_header':
                    section_headers.append(node)
    
    # Sort by ID (self_ref) in ascending order
    # Extract numeric part from ID for proper sorting (e.g., "#/texts/0" -> 0, "#/texts/10" -> 10)
    def extract_id_number(node):
        self_ref = node.get('self_ref', '')
        # Extract the last number from the self_ref (e.g., "#/texts/5" -> 5)
        import re
        numbers = re.findall(r'\d+', self_ref)
        if numbers:
            return int(numbers[-1])  # Use the last number in the ID
        return 0  # Default for nodes without numbers
    
    # Sort section headers by their ID number
    section_headers.sort(key=extract_id_number)
    
    return section_headers


def get_section_headers_by_page_order(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all nodes with label 'section_header' and return them in order of page and position.
    
    Args:
        tree: Tree structure containing the nodes
        
    Returns:
        List of section header nodes sorted by page number and position
    """
    # Define all node array types to search
    node_arrays = ['texts', 'tables', 'pictures', 'groups', 'key_value_items', 'form_items']
    
    section_headers = []
    
    # Search for section headers in all node arrays
    for array_name in node_arrays:
        if array_name in tree and tree[array_name]:
            for node in tree[array_name]:
                if node.get('label') == 'section_header':
                    section_headers.append(node)
    
    # Sort by page and position using the existing sorting function
    section_headers = sort_nodes_by_page_and_bbox(section_headers)
    
    return section_headers


def get_node_text_with_children(tree: Dict[str, Any], node_id: str) -> str:
    """
    Get the complete text for a node including its own text and all its children's text.
    Children are sorted by their ID order. Handles tables, images, and other non-text nodes appropriately.
    
    Args:
        tree: Tree structure containing the nodes
        node_id: The ID of the node (e.g., "#/texts/0", "#/tables/0", etc.)
        
    Returns:
        Complete text string including node's own text and all children's text
    """
    # Get the node information
    node_info = get_node_info(tree, node_id)
    if not node_info:
        return ""
    
    # Start with the node's own text
    complete_text = node_info.get('text', '')
    
    # Get children if they exist
    children = node_info.get('children', [])
    if not children:
        return complete_text
    
    # Sort children texts by their ID (extract numeric part for sorting)
    def extract_id_number(child_ref):
        child_id = child_ref.get('$ref', '')
        import re
        numbers = re.findall(r'\d+', child_id)
        if numbers:
            return int(numbers[-1])  # Use the last number in the ID
        return 0
    
    # Create a list of (child_ref, text) tuples for sorting
    child_ref_text_pairs = []
    for child_ref in children:
        child_id = child_ref.get('$ref', '')
        
        child_text = get_node_text(tree, child_id)
        
        # Include all children, even if they don't have text (like images, tables)
        # This ensures we don't lose information about the document structure
        if child_text:  # Include non-empty text and placeholders like [IMAGE], [TABLE]
            child_ref_text_pairs.append((child_ref, child_text))
    
    
    # Concatenate the sorted children texts
    if child_ref_text_pairs:
        children_text = ' '.join([text for _, text in child_ref_text_pairs])
        if complete_text.strip() and children_text.strip():
            complete_text = complete_text + ' ' + children_text
        elif children_text.strip():
            complete_text = children_text
    
    return complete_text


def get_section_header_texts(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get the complete text for each section header node including its own text and all children's text.
    
    Args:
        tree: Tree structure containing the nodes
        
    Returns:
        List of dictionaries containing section header information with complete text
    """
    section_headers = get_section_headers_by_id_order(tree)
    section_header_texts = []
    
    for section in section_headers:
        section_id = section.get('self_ref', '')
        section_text = section.get('text', '')
        
        # Get complete text including children
        complete_text = get_node_text_with_children(tree, section_id)
        
        # Get page number
        page_no = "?"
        if 'prov' in section and section['prov']:
            prov = section['prov'][0]
            page_no = str(prov.get('page_no', '?'))
        
        # Get children count
        children_count = len(section.get('children', []))
        
        section_info = {
            'id': section_id,
            'text': section_text,
            'complete_text': complete_text,
            'page': page_no,
            'children_count': children_count
        }
        
        section_header_texts.append(section_info)
    
    return section_header_texts


def get_table_details(tree: Dict[str, Any], table_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a table node, including its structure and content.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node (e.g., "#/tables/0")
        
    Returns:
        Dictionary containing detailed table information
    """
    # Find the table node
    if 'tables' in tree and tree['tables']:
        for table in tree['tables']:
            if table.get('self_ref') == table_id:
                table_info = {
                    'id': table_id,
                    'has_text': 'text' in table,
                    'text_content': table.get('text', ''),
                    'has_cells': 'cells' in table,
                    'cells_count': len(table.get('cells', [])),
                    'cells_content': [],
                    'raw_structure': table
                }
                
                # Extract cell information
                if 'cells' in table:
                    for i, cell in enumerate(table['cells']):
                        cell_info = {
                            'index': i,
                            'has_text': 'text' in cell if isinstance(cell, dict) else False,
                            'text': cell.get('text', '') if isinstance(cell, dict) else str(cell),
                            'type': type(cell).__name__,
                            'raw': cell
                        }
                        table_info['cells_content'].append(cell_info)
                
                return table_info
    
    return {'id': table_id, 'error': 'Table not found'}


def get_children_info(tree: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
    """
    Get information about all children of a node, including their types and IDs.
    
    Args:
        tree: Tree structure containing the nodes
        node_id: The ID of the node to get children for
        
    Returns:
        List of dictionaries containing child information
    """
    node_info = get_node_info(tree, node_id)
    if not node_info:
        return []
    
    children = node_info.get('children', [])
    children_info = []
    
    for child_ref in children:
        child_id = child_ref.get('$ref', '')
        print(f"node_id, child_id: {node_id}, {child_id}")
        if child_id:
            # Determine the node type from the ID
            node_type = "unknown"
            if child_id.startswith("#/texts/"):
                node_type = "text"
            elif child_id.startswith("#/tables/"):
                node_type = "table"
            elif child_id.startswith("#/pictures/"):
                node_type = "picture"
            elif child_id.startswith("#/groups/"):
                node_type = "group"
            elif child_id.startswith("#/key_value_items/"):
                node_type = "key_value_item"
            elif child_id.startswith("#/form_items/"):
                node_type = "form_item"
            
            # Get the child's text content
            child_text = get_node_text(tree, child_id)
            
            child_info = {
                'id': child_id,
                'type': node_type,
                'text': child_text[:100] + "..." if len(child_text) > 100 else child_text
            }
            children_info.append(child_info)
    
    return children_info


def get_table_text_by_bbox(tree: Dict[str, Any], table_id: str) -> str:
    """
    Get all text from a table by extracting text cell by cell, ordered by bounding box.
    Sorts by y-coordinate (top to bottom), then by x-coordinate (left to right).
    Also checks the 'data' field for table content.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node (e.g., "#/tables/0")
        
    Returns:
        String containing all table text ordered by position
    """
    # Find the table node
    if 'tables' in tree and tree['tables']:
        for table in tree['tables']:
            if table.get('self_ref') == table_id:
                # First, check if there's a 'data' field with table content
                if 'data' in table and table['data']:
                    data_content = table['data']
                    if isinstance(data_content, dict) and 'table_cells' in data_content:
                        # Extract text from table_cells, ordered by bounding box
                        table_cells = data_content['table_cells']
                        if table_cells:
                            # Sort cells by y-coordinate (top to bottom), then by x-coordinate (left to right)
                            # Note: coordinates are in BOTTOMLEFT format, so higher y values are higher on page
                            sorted_cells = sorted(table_cells, key=lambda cell: (
                                -cell['bbox']['t'],  # Negative because higher t values are higher on page
                                cell['bbox']['l']    # Left to right
                            ))
                            
                            # Extract text from sorted cells
                            text_parts = []
                            for cell in sorted_cells:
                                if 'text' in cell and cell['text'].strip():
                                    text_parts.append(cell['text'].strip())
                            
                            if text_parts:
                                return ' '.join(text_parts)
                    
                    # Fallback for other data formats
                    if isinstance(data_content, str):
                        return data_content
                    elif isinstance(data_content, list):
                        # If data is a list, try to extract text from it
                        text_parts = []
                        for item in data_content:
                            if isinstance(item, dict):
                                if 'text' in item:
                                    text_parts.append(item['text'])
                                elif 'content' in item:
                                    text_parts.append(str(item['content']))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        if text_parts:
                            return ' '.join(text_parts)
                    elif isinstance(data_content, dict):
                        # If data is a dict, try to extract text from it
                        if 'text' in data_content:
                            return data_content['text']
                        elif 'content' in data_content:
                            return str(data_content['content'])
                        else:
                            return str(data_content)
                
                # Get all children of the table
                children = table.get('children', [])
                if not children:
                    return "[EMPTY TABLE - NO CHILDREN]"
                
                # Collect all text nodes with their bounding box information
                text_cells = []
                
                for child_ref in children:
                    child_id = child_ref.get('$ref', '')
                    if child_id and child_id.startswith('#/texts/'):
                        # Get the text node
                        text_node = None
                        if 'texts' in tree and tree['texts']:
                            for text in tree['texts']:
                                if text.get('self_ref') == child_id:
                                    text_node = text
                                    break
                        
                        if text_node:
                            # Get text content
                            text_content = text_node.get('text', '').strip()
                            if text_content:
                                # Get bounding box from provenance
                                bbox = None
                                if 'prov' in text_node and text_node['prov']:
                                    prov = text_node['prov'][0]
                                    if 'bbox' in prov:
                                        bbox = prov['bbox']
                                
                                if bbox and len(bbox) >= 4:
                                    # bbox format: [x1, y1, x2, y2]
                                    x1, y1, x2, y2 = bbox[:4]
                                    text_cells.append({
                                        'text': text_content,
                                        'x1': x1,
                                        'y1': y1,
                                        'x2': x2,
                                        'y2': y2,
                                        'center_x': (x1 + x2) / 2,
                                        'center_y': (y1 + y2) / 2
                                    })
                                else:
                                    # If no bbox, add at the end
                                    text_cells.append({
                                        'text': text_content,
                                        'x1': 0,
                                        'y1': 0,
                                        'x2': 0,
                                        'y2': 0,
                                        'center_x': 0,
                                        'center_y': 0
                                    })
                
                if not text_cells:
                    return "[NO TEXT CELLS FOUND IN CHILDREN]"
                
                # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
                # Use center coordinates for more accurate sorting
                text_cells.sort(key=lambda cell: (cell['center_y'], cell['center_x']))
                
                # Join all text with spaces
                return ' '.join([cell['text'] for cell in text_cells])
    
    return "[TABLE NOT FOUND]"


def get_structured_table_content(tree: Dict[str, Any], table_id: str) -> List[Dict[str, Any]]:
    """
    Get structured table content with labels for column headers and table cells.
    Returns a list of dictionaries, each containing text, label, and position info.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node (e.g., "#/tables/0")
        
    Returns:
        List of dictionaries with keys: 'text', 'label', 'bbox', 'row_idx', 'col_idx'
        where label is either 'column_header' or 'table_cell'
    """
    # Find the table node
    if 'tables' in tree and tree['tables']:
        for table in tree['tables']:
            if table.get('self_ref') == table_id:
                # Check if there's a 'data' field with table content
                if 'data' in table and table['data']:
                    data_content = table['data']
                    if isinstance(data_content, dict) and 'table_cells' in data_content:
                        # Extract structured content from table_cells
                        table_cells = data_content['table_cells']
                        if table_cells:
                            # Sort cells by y-coordinate (top to bottom), then by x-coordinate (left to right)
                            sorted_cells = sorted(table_cells, key=lambda cell: (
                                -cell['bbox']['t'],  # Negative because higher t values are higher on page
                                cell['bbox']['l']    # Left to right
                            ))
                            
                            # Create structured content list
                            structured_content = []
                            for cell in sorted_cells:
                                if 'text' in cell and cell['text'].strip():
                                    # Determine label based on cell properties
                                    if cell.get('column_header', False):
                                        label = 'column_header'
                                    elif cell.get('row_header', False):
                                        label = 'row_header'
                                    elif cell.get('row_section', False):
                                        label = 'row_section'
                                    else:
                                        label = 'table_cell'
                                    
                                    structured_content.append({
                                        'text': cell['text'].strip(),
                                        'label': label,
                                        'bbox': cell['bbox'],
                                        'row_idx': cell.get('start_row_offset_idx', -1),
                                        'col_idx': cell.get('start_col_offset_idx', -1),
                                        'row_span': cell.get('row_span', 1),
                                        'col_span': cell.get('col_span', 1)
                                    })
                            
                            return structured_content
    
    return []


def get_plain_text_table(tree: Dict[str, Any], table_id: str) -> str:
    """
    Get plain text version of table content, sorted by row then column.
    Concatenates output line by line into a string.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node (e.g., "#/tables/0")
        
    Returns:
        String containing plain text table content, line by line
    """
    structured_content = get_structured_table_content(tree, table_id)
    
    if not structured_content:
        return "[NO TABLE CONTENT FOUND]"
    
    # Sort by row first, then by column
    sorted_content = sorted(structured_content, key=lambda item: (item['row_idx'], item['col_idx']))
    
    # Group by rows
    rows = {}
    for item in sorted_content:
        row_idx = item['row_idx']
        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append(item)
    
    # Build plain text output line by line
    text_lines = []
    for row_idx in sorted(rows.keys()):
        row_items = rows[row_idx]
        # Sort items within the row by column
        row_items.sort(key=lambda item: item['col_idx'])
        
        # Create line for this row
        line_parts = []
        for item in row_items:
            line_parts.append(item['text'])
        
        text_lines.append(' | '.join(line_parts))
    
    # Add prefix and suffix
    table_content = '\n'.join(text_lines)
    return f"The below is a table, table starts:\n{table_content}\nThis is the end of the table."



def print_plain_text_table(tree: Dict[str, Any], table_id: str) -> None:
    """
    Print plain text version of table content for easy inspection.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node to examine
    """
    print(f"\n=== Plain Text Table: {table_id} ===")
    
    plain_text = get_plain_text_table(tree, table_id)
    print(plain_text)


def print_structured_table_content(tree: Dict[str, Any], table_id: str) -> None:
    """
    Print structured table content with labels for easy inspection.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node to examine
    """
    print(f"\n=== Structured Table Content: {table_id} ===")
    
    structured_content = get_structured_table_content(tree, table_id)
    
    if not structured_content:
        print("No structured content found")
        return
    
    print(f"Found {len(structured_content)} table elements:")
    print("-" * 80)
    
    for i, item in enumerate(structured_content):
        print(f"{i+1:2d}. [{item['label']:15s}] Row:{item['row_idx']:2d} Col:{item['col_idx']:2d} | {item['text']}")
    
    # Group by label for summary
    label_counts = {}
    for item in structured_content:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nSummary by label:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} items")


def print_table_raw_content(tree: Dict[str, Any], table_id: str) -> None:
    """
    Print the raw content of a table node for inspection.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node to examine
    """
    print(f"\n=== Raw Table Content: {table_id} ===")
    
    # Find the table node
    if 'tables' in tree and tree['tables']:
        for table in tree['tables']:
            if table.get('self_ref') == table_id:
                print("Complete table structure:")
                print(json.dumps(table, indent=2, default=str))
                return
    
    print(f"Table {table_id} not found")


def print_table_text_details(tree: Dict[str, Any], table_id: str) -> None:
    """
    Print detailed text information from a table, showing cell-by-cell extraction.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node to examine
    """
    print(f"\n=== Table Text Details: {table_id} ===")
    
    # Find the table node
    if 'tables' in tree and tree['tables']:
        for table in tree['tables']:
            if table.get('self_ref') == table_id:
                # Get all children of the table
                children = table.get('children', [])
                print(f"Table has {len(children)} children")
                
                if not children:
                    print("No children found in table")
                    return
                
                # Collect all text nodes with their bounding box information
                text_cells = []
                
                for child_ref in children:
                    child_id = child_ref.get('$ref', '')
                    if child_id and child_id.startswith('#/texts/'):
                        # Get the text node
                        text_node = None
                        if 'texts' in tree and tree['texts']:
                            for text in tree['texts']:
                                if text.get('self_ref') == child_id:
                                    text_node = text
                                    break
                        
                        if text_node:
                            # Get text content
                            text_content = text_node.get('text', '').strip()
                            if text_content:
                                # Get bounding box from provenance
                                bbox = None
                                if 'prov' in text_node and text_node['prov']:
                                    prov = text_node['prov'][0]
                                    if 'bbox' in prov:
                                        bbox = prov['bbox']
                                
                                if bbox and len(bbox) >= 4:
                                    # bbox format: [x1, y1, x2, y2]
                                    x1, y1, x2, y2 = bbox[:4]
                                    text_cells.append({
                                        'id': child_id,
                                        'text': text_content,
                                        'x1': x1,
                                        'y1': y1,
                                        'x2': x2,
                                        'y2': y2,
                                        'center_x': (x1 + x2) / 2,
                                        'center_y': (y1 + y2) / 2
                                    })
                                else:
                                    # If no bbox, add at the end
                                    text_cells.append({
                                        'id': child_id,
                                        'text': text_content,
                                        'x1': 0,
                                        'y1': 0,
                                        'x2': 0,
                                        'y2': 0,
                                        'center_x': 0,
                                        'center_y': 0
                                    })
                
                if not text_cells:
                    print("No text cells found in table")
                    return
                
                # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
                text_cells.sort(key=lambda cell: (cell['center_y'], cell['center_x']))
                
                print(f"Found {len(text_cells)} text cells:")
                for i, cell in enumerate(text_cells):
                    print(f"  {i+1}. {cell['id']}: '{cell['text']}' at ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
                
                # Show the complete ordered text
                complete_text = ' '.join([cell['text'] for cell in text_cells])
                print(f"\nComplete table text (ordered by position):")
                print(f"'{complete_text}'")
                
                return
    
    print(f"Table {table_id} not found")


def print_table_details(tree: Dict[str, Any], table_id: str) -> None:
    """
    Print detailed information about a table node.
    
    Args:
        tree: Tree structure containing the nodes
        table_id: The ID of the table node to examine
    """
    table_info = get_table_details(tree, table_id)
    
    print(f"\n=== Table Details: {table_id} ===")
    
    if 'error' in table_info:
        print(f"Error: {table_info['error']}")
        return
    
    print(f"Table ID: {table_info['id']}")
    print(f"Has text field: {table_info['has_text']}")
    if table_info['has_text']:
        print(f"Text content: '{table_info['text_content']}'")
    
    print(f"Has cells: {table_info['has_cells']}")
    print(f"Number of cells: {table_info['cells_count']}")
    
    # Check for data field
    if 'data' in table_info['raw_structure'] and table_info['raw_structure']['data']:
        data_content = table_info['raw_structure']['data']
        print(f"Has data field: True")
        print(f"Data type: {type(data_content).__name__}")
        if isinstance(data_content, str):
            print(f"Data content: '{data_content[:200]}{'...' if len(data_content) > 200 else ''}'")
        elif isinstance(data_content, (list, dict)):
            print(f"Data content: {str(data_content)[:200]}{'...' if len(str(data_content)) > 200 else ''}")
        else:
            print(f"Data content: {data_content}")
    else:
        print("Has data field: False")
    
    if table_info['cells_content']:
        print("\nCell details:")
        for cell in table_info['cells_content']:
            print(f"  Cell {cell['index']}:")
            print(f"    Type: {cell['type']}")
            print(f"    Has text: {cell['has_text']}")
            if cell['has_text']:
                print(f"    Text: '{cell['text']}'")
            else:
                print(f"    Content: '{cell['text']}'")
    
    # Show the raw structure (first few keys)
    print(f"\nRaw structure keys: {list(table_info['raw_structure'].keys())}")
    
    # Show transformed text as it would appear in the complete text
    transformed_text = get_node_text(tree, table_id)
    print(f"\nTransformed text (as used in complete text): '{transformed_text}'")

def get_node_order(tree, node_id):
    """
    Given a tree and a section header `node_id`, return a tuple:
        (number_of_section_headers_before_it, total_number_of_section_headers)

    If the `node_id` is not found among section headers, returns (-1, total).
    """
    try:
        section_headers = get_section_headers_by_id_order(tree) or []
        total = len(section_headers)

        for idx, header in enumerate(section_headers):
            if header.get('self_ref') == node_id:
                return idx, total

        return -1, total
    except Exception:
        # In case the tree structure is unexpected
        return -1, 0



# Example usage
if __name__ == "__main__":
    data_folder = Path('/Users/yiminglin/Documents/Codebase/LSF/data/CUAD_v1/full_contract_pdf')
    
    # Scan and collect all PDFs from the data folder
    input_doc_paths = []
    for pdf_file in data_folder.rglob("*.pdf"):
        input_doc_paths.append(pdf_file)

    i = 0
    for doc_file in input_doc_paths:
        i += 1
        input_path = Path(doc_file)
        doc_filename = input_path.stem  # Get filename without extension
    
        input_str = str(input_path)
        output_str = input_str.replace('/data/', '/out/')
        output_path = Path(output_str)
        output_dir = output_path.parent / doc_filename
        
        json_file = output_dir / f"{doc_filename}.json"
        tree = process_json_file(str(json_file))

        print_tree_structure(tree) 

        idx, total = get_node_order(tree, "#/texts/230")
        text = get_node_text_only(tree, "#/texts/230")
        page_no, total_pages = get_node_page_info(tree, "#/texts/230")
        print(idx, total)
        print(text)
        print(page_no, total_pages)
        break 



    
