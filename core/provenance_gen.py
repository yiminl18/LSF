import os
import sys,json

from pyarrow import NULL

# Add the current directory to the path so we can import from other modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# PDF text extraction
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PyPDF2 not available. Install with: pip install PyPDF2")

# Import functions with error handling

from tree_gen import get_node_text_with_children, get_section_headers_by_id_order, process_json_file, get_node_order, get_node_text_only, get_node_page_info
from model import model
from data_ingestion import read_master_clauses_csv, read_questions_txt,read_pdfs_from_folder, get_ans


def extract_pdf_text(pdf_path):
    """
    Extract raw text from a PDF document.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        str: Extracted text content, or None if extraction fails
    """
    if not PDF_AVAILABLE:
        print("Error: PyPDF2 not available. Cannot extract PDF text.")
        return None
    
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text_content = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
            
            return text_content.strip()
            
    except FileNotFoundError:
        print(f"Error: PDF file not found: {pdf_path}")
        return None
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return None


def extract_pdf_text_simple(pdf_path, output_path=None):
    """
    Simple function to extract text from PDF and optionally save to file.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save extracted text. If None, returns text only.
    
    Returns:
        str: Extracted text content, or None if extraction fails
    """
    text = extract_pdf_text(pdf_path)
    
    if text and output_path:
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text saved to: {output_path}")
        except Exception as e:
            print(f"Error saving text to file: {e}")
    
    return text

def extract_raw_texts():
    """
    Extract raw text from all PDFs and save to corresponding text files.
    """
    pdf_folder = '/Users/yiminglin/Documents/Codebase/LSF/data/CUAD_v1/full_contract_pdf/'
    pdf_files = read_pdfs_from_folder(pdf_folder)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for i, pdf_info in enumerate(pdf_files):
        pdf_path = pdf_info['full_path']
        doc_name = pdf_info['filename']
        
        # Create output path by replacing /full_contract_pdf/ with /full_contract_raw_txt/
        output_path = pdf_path.replace('/full_contract_pdf/', '/full_contract_raw_txt/').replace('.pdf', '.txt')
        
        # Extract and save text
        text = extract_pdf_text_simple(pdf_path, output_path)
        
        if text:
            print(f"✅ Successfully processed: {doc_name}")
        else:
            print(f"❌ Failed to process: {doc_name}")
    
    print(f"\n=== Text extraction completed ===")



def semantic_equivalence_checker(answer1, answer2, model_name='gpt_41_mini_azure'):
    prompt = [
        f"Are these two answers semantically equivalent? Answer only 'YES' or 'NO'.\n\nAnswer 1: {answer1}\n\nAnswer 2: {answer2}",
        ""
    ]
    
    try:
        response = model(prompt, model_name)
        return response.strip().upper() == "YES"
    except Exception as e:
        print(f"Error in semantic equivalence check: {e}")
        return False


def QA(question, context, model_name='gpt_41_mini_azure'):
    # Create prompt for LLM
    prompt = (f"Question: {question}. Only return the answer based on the given text. Do not add any explanations. \n\nText: ", context) 

    # Get LLM response
    llm_answer = model(prompt, model_name)
    return llm_answer

def create_summary(node_text, answer, question, model_name='gpt_41_mini_azure'):
    """
    Very simple summary helper. Splits node_text into sentences, asks LLM to return
    the sentence IDs that support the given answer for the question, and returns
    a concatenation of those sentences.
    """
    import re

    if not node_text:
        return ""

    # Split into rough sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', node_text) if s and s.strip()]
    if not sentences:
        sentences = [node_text.strip()]

    # Build context as lines of "id: sentence"
    indexed_context_lines = [f"{i+1}: {sent}" for i, sent in enumerate(sentences)]
    context_block = "\n".join(indexed_context_lines)

    # Prompt: ask for sentence IDs only
    prompt = (
        f"Given the question and answer, return ONLY the sentence IDs (numbers) that provide the answer to the question. "
        f"Respond as a comma-separated list of numbers without explanations.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Context (sentence_id: sentence):\n{context_block}\n"
    )

    try:
        response = model((prompt,''), model_name)
    except Exception as e:
        print(f"create_summary LLM error: {e}")
        return ""

    print(f"create_summary response: {response}")

    # Parse IDs from response
    ids = set()
    for token in re.findall(r"\d+", response or ""):
        try:
            idx = int(token) - 1
            if 0 <= idx < len(sentences):
                ids.add(idx)
        except Exception:
            continue

    if not ids:
        return ""
    
    print(f"create_summary ids: {ids}")

    # Preserve original order
    chosen_sentences = [sentences[i] for i in sorted(ids)]
    summary = " ".join(chosen_sentences)
    return summary

def sanitize_question_category(question_category):
    if not question_category:
        return question_category
    return question_category.replace('/', '_')

def provenance_gen(question, answer, tree, model_name='gpt_41_mini_azure'):
    print('question', question)
    provenance_nodes = []
    
    # Get all section headers using the tree_gen function
    section_headers = get_section_headers_by_id_order(tree)
    
    for section_header in section_headers:
        node_id = section_header.get('self_ref', '')
        if not node_id:
            continue
        
        try:
            # Get the text content of this section header node
            node_text = get_node_text_with_children(tree, node_id)
            # print(node_id)
            # print(f"Node text: {node_text[:300]}{'...' if len(node_text) > 300 else ''}")
            
            if not node_text or node_text.strip() == "":
                continue

            # Get LLM response
            llm_answer = QA(question, node_text, model_name)
            
            if not llm_answer or llm_answer.strip() == "":
                continue
            
            # Check semantic equivalence
            is_equivalent = semantic_equivalence_checker(answer, llm_answer, model_name)

            #print(f"Is equivalent: {is_equivalent}")
            
            if is_equivalent:
                provenance_nodes.append({
                    'id': node_id,
                    'ans': llm_answer
                })
                #print(f"✅ Found matching section header: {node_id}")
            
        except Exception as e:
            print(f"Error processing section header {node_id}: {e}")
            continue
    
    print(f"Found {len(provenance_nodes)} section headers with semantically equivalent answers")
    return provenance_nodes





def group_pdfs_by_folder(pdf_folder_path):
    """
    Group PDFs by their most fine-grained folder (immediate parent folder).
    
    Args:
        pdf_folder_path (str): Path to the root folder containing PDF files
    
    Returns:
        list: List of dictionaries, each containing:
              - 'root_folder_path': The immediate parent folder path
              - 'pdf_paths': List of PDF file paths in that folder
    """
    import glob
    
    pdf_groups = {}
    
    # Find all PDF files recursively
    pdf_pattern = os.path.join(pdf_folder_path, "**", "*.pdf")
    pdf_files = glob.glob(pdf_pattern, recursive=True)
    
    # Also search for uppercase PDF extension
    pdf_pattern_upper = os.path.join(pdf_folder_path, "**", "*.PDF")
    pdf_files.extend(glob.glob(pdf_pattern_upper, recursive=True))
    
    # Remove duplicates
    pdf_files = list(set(pdf_files))
    
    # Group PDFs by their immediate parent folder
    for pdf_path in pdf_files:
        # Get the immediate parent folder
        parent_folder = os.path.dirname(pdf_path)
        
        # Initialize the group if it doesn't exist
        if parent_folder not in pdf_groups:
            pdf_groups[parent_folder] = []
        
        # Add the PDF to its group
        pdf_groups[parent_folder].append(pdf_path)
    
    # Convert to list of dictionaries
    result = []
    for folder_path, pdf_paths in pdf_groups.items():
        result.append({
            'root_folder_path': folder_path,
            'pdf_paths': sorted(pdf_paths)  # Sort for consistent ordering
        })
    
    # Sort by folder path for consistent ordering
    result.sort(key=lambda x: x['root_folder_path'])
    
    return result


def get_tree_path(doc_path):
    # Convert PDF path to JSON path
    # /data/CUAD_v1/full_contract_pdf/Part_I/Co_Branding/file.pdf
    # -> /out/CUAD_v1/full_contract_pdf/Part_I/Co_Branding/file/file.json
    json_path = doc_path.replace('/data/', '/out/').replace('.pdf', '.json')
    
    # Check if the JSON file exists directly
    if os.path.exists(json_path):
        return json_path
    
    # If not, try the subdirectory structure
    # Remove .json extension and add subdirectory
    base_path = json_path.replace('.json', '')
    subdir_path = os.path.join(base_path, os.path.basename(base_path) + '.json')
    
    if os.path.exists(subdir_path):
        return subdir_path
    
    return json_path  # Return original path if neither exists


def get_raw_text_path(pdf_path):
    return pdf_path.replace('/full_contract_pdf/', '/full_contract_raw_txt/').replace('.pdf', '.txt')

def read_text_file(text_path):
    with open(text_path, 'r') as f:
        return f.read()

def read_json_file(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"JSON file not found: {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_path}: {e}")
        return None
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
        return None


def test_provenance():
    import json
    
    master_clauses_data = read_master_clauses_csv()
    questions = read_questions_txt()
    pdf_folder = '/Users/yiminglin/Documents/Codebase/LSF/data/CUAD_v1/full_contract_pdf/'
    
    # Group PDFs by their immediate parent folder
    pdf_groups = group_pdfs_by_folder(pdf_folder)
    print(f"Found {len(pdf_groups)} folder groups")
    
    # Create output directory for results
    raw_output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'provenance_results')

    #print('output_dir_init', output_dir)
    
    # Process each folder group
    for group_idx, group in enumerate(pdf_groups):  # scan group 
        folder_path = group['root_folder_path']
        pdf_paths = group['pdf_paths']
        folder_name = os.path.basename(folder_path)
        print('folder_name', folder_name)
        output_dir = os.path.join(raw_output_dir, folder_name) 
        print('output_dir', output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Processing Folder {group_idx + 1}: {folder_name} ===")
        print(f"PDFs in this folder: {len(pdf_paths)}")
        
        # Process each question for this document
        for question in questions:  # scan question 
            group_question_results = []
            # Check if results file already exists for this group
            question_text = question['description']
            question_category = question['category']
            question_category = sanitize_question_category(question_category)

            output_file = os.path.join(output_dir, f'{question_category}_provenance_results.json')

            print(f"output_file: {output_file}")
            print('question_category', question_category)

            if os.path.exists(output_file):
                continue

            
            print(f"  Question: {question_text}")
        
            # Process each PDF in this folder
            for pdf_path in pdf_paths:
                doc_name = os.path.basename(pdf_path)
                answer = get_ans(doc_name, question_category, master_clauses_data)
                tree_path = get_tree_path(pdf_path)
                print(f"\nProcessing document: {doc_name}")
                tree = process_json_file(tree_path)   
                raw_text = read_text_file(get_raw_text_path(pdf_path)) 
                global_answer = QA(question_text, raw_text)  

                if answer['answer'] is None or answer['answer'] == NULL:
                    print('answer is None')
                    provenance_nodes = []
                else:
                    provenance_nodes = provenance_gen(question_text, answer['answer'], tree)
                
                print(f"    Number of provenance nodes: {len(provenance_nodes)}")
                
                # Collect node details
                node_details = []
                for node in provenance_nodes:
                    node_text = get_node_text_with_children(tree, node['id'])
                    idx, total = get_node_order(tree, node['id'])
                    node_summary = create_summary(node_text, answer['answer'], question_text)
                    header_text = get_node_text_only(tree, node['id'])
                    page_no, total_pages = get_node_page_info(tree, node['id'])
                    node_details.append({
                        'node_id': node['id'],
                        'node_text': node_text,
                        'llm_answer': node['ans'],
                        'node_summary': node_summary,
                        'header_text': header_text,
                        'page_no': page_no,
                        'total_pages': total_pages,
                        'tree_node': idx,
                        'tree_header_num': total
                    })
                    
                
                doc_result = {
                    'doc_path': pdf_path,
                    'question_text': question_text,
                    'question_category': question_category,
                    'true_answer': answer['answer'],
                    'llm_global_answer': global_answer,
                    'provenance_nodes': node_details
                }

                group_question_results.append(doc_result) 
        
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(group_question_results, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Results saved to: {output_file}")
            except Exception as e:
                print(f"❌ Error saving results for {folder_name}: {e}")
            
        
        #break 
    
    print(f"\n=== All results saved to: {output_dir} ===") 


if __name__ == "__main__":
    
    print("=== Running Provenance Generation ===")
    test_provenance()
