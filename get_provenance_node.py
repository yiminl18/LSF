import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from map_lsf_to_docling import process_pdf_with_both_tools
from calculate_similarity import calculate_similarity
from ask import ask
from evaluate_baseline import equal_llm, normalize_exact
from gpt_4o_azure import gpt_4o_azure

PROMPT_TEMPLATE = (
    "You are an expert in document structure.\n"
    "The document has been extracted as a FLAT list of headers (no real tree): "
    "all headers are in one list and the 'level' field may NOT reflect the true hierarchy.\n"
    "Each header has JSON metadata including: orig, text, summary, font, size, bold, all_cap, "
    "is_center, page_no, bbox, etc. The summary/text may span from this header until the next header, "
    "so boundaries are fuzzy.\n\n"

    "Task: given two headers X and Y from the SAME document, decide if X is the DIRECT parent of Y "
    "in the true section hierarchy. DIRECT parent means Y is an immediate subsection of X "
    "(no other header between them in the hierarchy).\n\n"

    "First, check order: X must appear BEFORE Y in reading order (by page_no and then vertical position "
    "from bbox). If Y appears before X, you MUST answer 'no'.\n\n"

    "Then compare X and Y on EXACTLY THREE aspects:\n"
    "1) HEADER (names and numbering):\n"
    "   - Look at 'orig' and 'text'. For a parent→child pair, the child's title is usually a more specific\n"
    "     or numbered subsection of the parent (e.g. '3' → '3.1', '3.2'; 'Attention' → 'Scaled Dot-Product Attention').\n"
    "   - header_support = yes only if Y's header clearly looks like a direct subsection or refinement of X's header.\n\n"
    "2) VISUAL STYLE:\n"
    "   - Compare font, size, bold, all_cap, is_center. Parent headers are usually more visually prominent:\n"
    "     larger font size and/or stronger styling (bold, all caps, centered) than their children.\n"
    "   - visual_support = yes only if X clearly looks like a higher-level heading than Y.\n\n"
    "3) SUMMARY SEMANTICS:\n"
    "   - Compare summaries. A child section usually describes a more specific part or subtopic of what the parent\n"
    "     describes, not a different or unrelated topic.\n"
    "   - summary_support = yes only if Y clearly reads as a specific subtopic of X, despite noisy boundaries.\n\n"
    "Decision rule:\n"
    "- If X does not appear before Y in the document, answer 'no'.\n"
    "- Otherwise, evaluate header_support, visual_support, and summary_support.\n"
    "- Answer 'yes' ONLY if AT LEAST TWO of these three supports are clearly 'yes'.\n"
    "- If the evidence is weak, mixed, or ambiguous for an aspect, treat that aspect as NOT supporting X being\n"
    "  the parent. When in doubt, prefer answering 'no'.\n\n"

    "Header X JSON:\n{header_x_json}\n\n"
    "Header Y JSON:\n{header_y_json}\n\n"
    "Question: Is X the direct parent of Y in the true section hierarchy?\n"
    "Answer with EXACTLY one word, lowercase: yes or no."
)


def ask_if_parent(header_x: Dict[str, Any],
                  header_y: Dict[str, Any],
                  key_path: str = '/Users/evier/Documents/gpt-4o.txt') -> bool:
    """
    Determine if header_x is the direct parent of header_y.
    
    Args:
        header_x: JSON information of the first header
        header_y: JSON information of the second header
        key_path: Path to the API key file (default: '/Users/evier/Documents/gpt-4o.txt')
    
    Returns:
        True if header_x is the direct parent of header_y, False otherwise
    """
    # Create copies, remove text_span to reduce overhead
    header_x_copy = {k: v for k, v in header_x.items() if k != 'text_span'}
    header_y_copy = {k: v for k, v in header_y.items() if k != 'text_span'}
    
    # Format prompt
    header_x_json = json.dumps(header_x_copy, ensure_ascii=False, indent=2)
    header_y_json = json.dumps(header_y_copy, ensure_ascii=False, indent=2)
    
    prompt = PROMPT_TEMPLATE.format(
        header_x_json=header_x_json,
        header_y_json=header_y_json
    )
    
    # Call GPT-4o
    response = gpt_4o_azure(prompt, key_path=key_path, max_tokens=10, temperature=0)
    
    # Parse response (should be "yes" or "no")
    response_lower = response.strip().lower()
    
    # Check if contains "yes"
    if 'yes' in response_lower:
        return True
    elif 'no' in response_lower:
        return False
    else:
        # If response format is unexpected, default to False
        print(f"Warning: Unexpected response format: {response}, defaulting to False")
        return False


def get_leaf(pdf_path: str,
            question: str,
            answer: str,
            output_dir: Optional[str] = None,
            embedding_key_path: str = '/Users/evier/Documents/embedding_key.txt',
            gpt_key_path: str = '/Users/evier/Documents/gpt-4o.txt') -> Optional[Dict[str, Any]]:
    """
    Find the first header that can answer the question (sorted by similarity).
    
    Args:
        pdf_path: Path to the PDF file
        question: Question string
        answer: Expected answer string (for validation)
        output_dir: Directory for output files (default: 'result')
        embedding_key_path: Path to the embedding API key file
        gpt_key_path: Path to the GPT API key file
    
    Returns:
        Matching header dictionary, or None if not found
    """
    # 1. Call map_lsf_to_docling to get merged_json
    print(f"Processing PDF: {pdf_path}")
    merged_json_path = process_pdf_with_both_tools(pdf_path, output_dir=output_dir)
    
    # 2. Load merged_json
    with open(merged_json_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    headers = merged_data.get('texts', [])
    if not headers:
        print("No headers found")
        return None
    
    print(f"Found {len(headers)} headers")
    
    # 3. Calculate similarity between each header and question
    header_similarities = []
    total_headers = len(headers)
    valid_headers = 0
    
    print(f"\nStarting embedding similarity calculation...")
    for idx, header in enumerate(headers, 1):
        header_text = header.get('text', '')
        text_span = header.get('text_span', '')
        
        # Concatenate header name and text_span
        combined_text = f"{header_text} {text_span}".strip()
        
        if not combined_text:
            continue
        
        valid_headers += 1
        
        # Show progress
        print(f"[Progress] {valid_headers}/{total_headers} ({idx}/{total_headers}) - {header_text[:50]}...")
        
        # Calculate similarity
        similarity = calculate_similarity(combined_text, question, key_path=embedding_key_path)
        header_similarities.append((similarity, header, combined_text))
        print(f"      Similarity: {similarity:.4f}")
    
    print(f"\nComplete! Calculated similarity for {valid_headers} headers")
    
    # 4. Sort by similarity (descending)
    header_similarities.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Sorting by similarity complete, highest similarity: {header_similarities[0][0]:.4f}" if header_similarities else "No available headers")
    
    # 5. Iterate from highest to lowest similarity, find first header with correct answer
    for similarity, header, combined_text in header_similarities:
        print(f"\nTrying header: {header.get('text', '')[:50]}... (similarity: {similarity:.4f})")
        
        # Use ask function to get answer
        predicted_answer = ask(combined_text, question, key_path=gpt_key_path)
        
        # Check if answer is correct
        # First text comparison
        is_exact_match = normalize_exact(predicted_answer) == normalize_exact(answer)
        
        if is_exact_match:
            print(f"✓ Text match successful!")
            return header
        
        # If text doesn't match, use LLM to judge
        print(f"Text doesn't match, using LLM to judge...")
        is_equivalent, judge_tokens = equal_llm(predicted_answer, answer, question)
        
        if is_equivalent:
            print(f"✓ LLM judged as equivalent!")
            return header
        
        print(f"✗ Answer doesn't match")
    
    print("\nNo matching header found")
    return None


def get_parent(merged_json_path: str,
              header: Dict[str, Any],
              key_path: str = '/Users/evier/Documents/gpt-4o.txt') -> Optional[Dict[str, Any]]:
    """
    Find the direct parent of the given header.
    
    Args:
        merged_json_path: Path to merged JSON file
        header: Header dictionary to find parent for
        key_path: Path to the API key file (default: '/Users/evier/Documents/gpt-4o.txt')
    
    Returns:
        Direct parent header dictionary, or None if not found
    """
    # Load merged JSON
    with open(merged_json_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    headers = merged_data.get('texts', [])
    if not headers:
        return None
    
    # Find current header's position in the list
    current_header_idx = None
    for idx, h in enumerate(headers):
        # Match by self_ref or text
        if h.get('self_ref') == header.get('self_ref') or h.get('text') == header.get('text'):
            current_header_idx = idx
            break
    
    if current_header_idx is None:
        print("Warning: Header not found in merged JSON")
        return None
    
    # Search backwards from current position
    for i in range(current_header_idx - 1, -1, -1):
        candidate_parent = headers[i]
        print(f"Checking header {i}: {candidate_parent.get('text', '')[:50]}...")
        
        # Use ask_if_parent to determine
        if ask_if_parent(candidate_parent, header, key_path=key_path):
            print(f"Found direct parent: {candidate_parent.get('text', '')}")
            return candidate_parent
    
    print("No direct parent found")
    return None


def get_parent_path_str(merged_json_path: str,
                       header: Dict[str, Any],
                       key_path: str = '/Users/evier/Documents/gpt-4o.txt') -> str:
    """
    Get path string from the topmost ancestor to the current header (including header names and summaries).
    
    Args:
        merged_json_path: Path to merged JSON file
        header: Target header dictionary
        key_path: Path to the API key file (default: '/Users/evier/Documents/gpt-4o.txt')
    
    Returns:
        Path string in format "ancestor1_name ancestor1_summary ancestor2_name ancestor2_summary ... current_header_name current_header_summary"
    """
    # Collect all ancestors (including self)
    path = [header]
    current = header
    
    # Recursively find all ancestors
    while True:
        parent = get_parent(merged_json_path, current, key_path=key_path)
        if parent is None:
            break
        path.insert(0, parent)  # Insert at the beginning
        current = parent
    
    # Concatenate path string
    path_parts = []
    for h in path:
        header_text = h.get('text', '')
        summary = h.get('summary', '')
        
        if header_text:
            path_parts.append(header_text)
        if summary:
            path_parts.append(summary)
    
    return ' '.join(path_parts)


def construct_tree(merged_json_path: str,
                  root_text: str = 'Document Root',
                  key_path: str = '/Users/evier/Documents/gpt-4o.txt') -> Dict[str, Any]:
    """
    Build tree structure from merged JSON.
    
    Headers in merged_json are in pre-order traversal, use DFS stack to build tree.
    Root node defaults to represent the entire document.
    
    Args:
        merged_json_path: Path to merged JSON file
        root_text: Text for root node (default: 'Document Root')
        key_path: Path to the API key file (default: '/Users/evier/Documents/gpt-4o.txt')
    
    Returns:
        Root node dictionary of the tree, containing 'node' (header data) and 'children' (list of child nodes)
    """
    # Load merged JSON
    with open(merged_json_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    headers = merged_data.get('texts', [])
    if not headers:
        return None
    
    # Sort headers by page and position (ensure document reading order)
    def get_header_sort_key(header):
        """Get sort key: first by page number, then by y coordinate (top to bottom)."""
        prov = header.get('prov', [])
        if not prov:
            return (0, 0)  # If no prov, put at the beginning
        
        prov_item = prov[0]  # Take the first prov
        page_no = prov_item.get('page_no', 0)
        bbox = prov_item.get('bbox', {})
        
        # BOTTOMLEFT coordinate system: larger t values mean higher on page
        # Use t value as y coordinate (descending sort, larger t first)
        t_value = bbox.get('t', 0)
        
        return (page_no, -t_value)  # Negative sign for descending sort
    
    headers.sort(key=get_header_sort_key)
    
    # Create virtual root node representing the entire document
    root = {
        'node': {
            'text': root_text,
            'summary': None,
            'level': 0
        },
        'children': []
    }
    
    # Store all nodes for quick lookup (using self_ref as key)
    node_map = {}  # self_ref -> tree_node mapping
    
    print(f"Starting tree construction, {len(headers)} nodes...")
    print(f"Root node: {root_text}")
    
    for idx, header in enumerate(headers):
        current_node = {
            'node': header,
            'children': []
        }
        
        # From headers before current header, search backwards for first direct parent
        parent_node = None
        
        # Iterate backwards through previous headers
        for i in range(idx - 1, -1, -1):
            candidate_header = headers[i]
            candidate_self_ref = candidate_header.get('self_ref', '')
            candidate_node = node_map.get(candidate_self_ref)
            
            if candidate_node is None:
                continue
            
            # Check if it's a direct parent
            if ask_if_parent(candidate_header, header, key_path=key_path):
                parent_node = candidate_node
                print(f"[{idx+1}/{len(headers)}] Adding child node: {header.get('text', '')[:50]}... (parent: {candidate_header.get('text', '')[:30]}...)")
                break
        
        # If parent found, add to parent's children list
        if parent_node:
            parent_node['children'].append(current_node)
        else:
            # No parent found, add as child of root node
            root['children'].append(current_node)
            print(f"[{idx+1}/{len(headers)}] Adding child node: {header.get('text', '')[:50]}... (parent: {root_text})")
        
        # Add current node to mapping (using self_ref as key)
        header_self_ref = header.get('self_ref', f'#/texts/{idx}')
        node_map[header_self_ref] = current_node
    
    print(f"\nTree construction complete!")
    return root


def print_tree(tree_node: Dict[str, Any], indent: int = 0, max_depth: int = 3, current_depth: int = 0):
    """
    Recursively print tree structure.
    
    Args:
        tree_node: Tree node dictionary containing 'node' and 'children'
        indent: Current indentation level
        max_depth: Maximum print depth
        current_depth: Current depth
    """
    if current_depth >= max_depth:
        return
    
    prefix = "  " * indent + "├── " if indent > 0 else ""
    node_text = tree_node['node'].get('text', '')[:50]
    print(f"{prefix}{node_text}")
    
    for child in tree_node['children']:
        print_tree(child, indent + 1, max_depth, current_depth + 1)


def save_tree_to_file(tree_node: Dict[str, Any], output_path: str):
    """
    Save the entire tree to a file.
    
    Args:
        tree_node: Tree node dictionary containing 'node' and 'children'
        output_path: Output file path
    """
    def write_tree(node, file, indent=0, current_depth=0):
        """Recursively write tree structure to file"""
        prefix = "  " * indent + "├── " if indent > 0 else ""
        node_text = node['node'].get('text', '')
        summary = node['node'].get('summary', '')
        
        # Write node text
        file.write(f"{prefix}{node_text}\n")
        
        # If there's a summary, write it too
        if summary:
            summary_prefix = "  " * (indent + 1) + "└─ Summary: "
            file.write(f"{summary_prefix}{summary[:200]}...\n" if len(summary) > 200 else f"{summary_prefix}{summary}\n")
        
        # Recursively write child nodes
        for child in node['children']:
            write_tree(child, file, indent + 1, current_depth + 1)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        write_tree(tree_node, f)
    
    print(f"Tree structure saved to: {output_file}")


if __name__ == "__main__":
    # Build tree
    pdf_path = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
    output_dir = "result"
    
    # Get merged_json_path
    merged_json_path = process_pdf_with_both_tools(pdf_path, output_dir=output_dir)
    
    # Build tree
    print("\n" + "="*80)
    print("Building tree structure")
    print("="*80)
    tree = construct_tree(str(merged_json_path), root_text='Document Root')
    
    if tree:
        print("\n" + "="*80)
        print("Tree structure construction complete!")
        print("="*80)
        print(f"Root node: {tree['node'].get('text', '')}")
        print(f"Number of children: {len(tree['children'])}")
        
        # Print tree structure (first 3 levels)
        print("\nTree structure preview (first 3 levels):")
        print_tree(tree, max_depth=3)
        
        # Save entire tree to file
        pdf_file = Path(pdf_path)
        tree_output_path = Path(output_dir) / f"{pdf_file.stem}_tree.txt"
        save_tree_to_file(tree, str(tree_output_path))
    else:
        print("Tree construction failed")
    
    # 测试 ask_if_parent 的代码（注释掉）
    # """
    # merged_json_path = "result/NIPS-2017-attention-is-all-you-need-Paper_merged.json"
    # 
    # with open(merged_json_path, 'r', encoding='utf-8') as f:
    #     merged_data = json.load(f)
    # 
    # headers = merged_data.get('texts', [])
    # 
    # # 找到 "3 Model Architecture" 和 "3.2 Attention" header
    # header_3 = None
    # header_3_2 = None
    # 
    # for header in headers:
    #     text = header.get('text', '')
    #     if text == "3 Model Architecture":
    #         header_3 = header
    #     if text == "3.2 Attention":
    #         header_3_2 = header
    # 
    # if header_3 and header_3_2:
    #     print("="*80)
    #     print("测试 ask_if_parent")
    #     print("="*80)
    #     print(f"Header 3: {header_3.get('text', '')}")
    #     print(f"Header 3.2: {header_3_2.get('text', '')}")
    #     print("\n调用 ask_if_parent(header_3, header_3_2)...")
    #     result = ask_if_parent(header_3, header_3_2)
    #     print(f"\n结果: {result}")
    #     print(f"Header 3 {'是' if result else '不是'} Header 3.2 的直接父亲")
    # else:
    #     print("未找到指定的 headers")
    #     if not header_3:
    #         print("未找到 Header 3")
    #     if not header_3_2:
    #         print("未找到 Header 3.2")
    #     # 打印前10个 headers 的文本，帮助定位
    #     print("\n前10个 headers:")
    #     for i, h in enumerate(headers[:10]):
    #         print(f"  [{i}] {h.get('text', '')[:80]}")
    # """
    
    # 原有的 get_leaf 示例代码（注释掉）
    # """
    # question = "Did management report any material weaknesses in internal control over financial reporting (Yes/No)?"
    # answer = "No"
    # 
    # result = get_leaf(pdf_path, question, answer)
    # if result:
    #     print(f"\n找到匹配的 header: {result.get('text', '')}")
    #     
    #     # 获取直接父亲
    #     print(f"\n查找直接父亲...")
    #     parent = get_parent(str(merged_json_path), result)
    #     if parent:
    #         print(f"直接父亲: {parent.get('text', '')}")
    #     else:
    #         print("未找到直接父亲")
    #     
    #     # 获取完整路径字符串
    #     print(f"\n获取完整路径字符串...")
    #     path_str = get_parent_path_str(str(merged_json_path), result)
    #     print(f"完整路径:\n{path_str}")
    # else:
    #     print("\n未找到匹配的 header")
    # """

