import csv
import json
import os
import glob


def read_master_clauses_csv():
    """
    Read master_clauses.csv and convert it to a JSON list of dictionaries.
    
    Returns:
        list: A list of dictionaries, where each dictionary represents a row
              with column names as keys and cell values as values.
    """
    # Path to the master_clauses.csv file
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1', 'master_clauses.csv')
    
    # List to store the converted data
    data_list = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            # Create CSV reader
            csv_reader = csv.DictReader(csvfile)
            
            # Convert each row to a dictionary and add to the list
            for row in csv_reader:
                data_list.append(dict(row))
                
        print(f"Successfully read {len(data_list)} rows from master_clauses.csv")
        return data_list
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return []


def save_master_clauses_json(output_path=None):
    """
    Read master_clauses.csv and save it as a JSON file.
    
    Args:
        output_path (str, optional): Path where to save the JSON file.
                                   If None, saves to the same directory as the CSV.
    
    Returns:
        str: Path to the saved JSON file, or None if failed.
    """
    # Read the data
    data_list = read_master_clauses_csv()
    
    if not data_list:
        return None
    
    # Determine output path
    if output_path is None:
        csv_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1')
        output_path = os.path.join(csv_dir, 'master_clauses.json')
    
    try:
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data_list, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved JSON to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")
        return None


def read_questions_txt():
    """
    Read question.txt and convert it to a list of dictionaries.
    
    Each question entry contains:
    - id: Question number
    - category: The category name
    - description: Description of the question
    - answer_format: Expected answer format
    - group: Group number (or '-' if no group)
    
    Returns:
        list: A list of dictionaries, where each dictionary represents a question
              with keys: id, category, description, answer_format, group
    """
    # Path to the question.txt file
    txt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1', 'question.txt')
    
    # List to store the parsed questions
    questions_list = []
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as txtfile:
            lines = txtfile.readlines()
        
        current_question = {}
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if line starts with a number (question ID)
            if line.isdigit():
                # If we have a previous question, add it to the list
                if current_question:
                    questions_list.append(current_question)
                
                # Start new question
                current_question = {'id': int(line)}
            
            # Parse category
            elif line.startswith('Category:'):
                current_question['category'] = line.split('Category:')[1].strip()
            
            # Parse description
            elif line.startswith('Description:'):
                current_question['description'] = line.split('Description:')[1].strip()
            
            # Parse answer format
            elif line.startswith('Answer Format:'):
                current_question['answer_format'] = line.split('Answer Format:')[1].strip()
            
            # Parse group
            elif line.startswith('Group:'):
                group_value = line.split('Group:')[1].strip()
                # Convert group to integer if it's a number, otherwise keep as string
                if group_value.isdigit():
                    current_question['group'] = int(group_value)
                else:
                    current_question['group'] = group_value
        
        # Add the last question if it exists
        if current_question:
            questions_list.append(current_question)
        
        print(f"Successfully read {len(questions_list)} questions from question.txt")
        return questions_list
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {txt_path}")
        return []
    except Exception as e:
        print(f"Error reading question.txt file: {str(e)}")
        return []


def save_questions_json(output_path=None):
    """
    Read question.txt and save it as a JSON file.
    
    Args:
        output_path (str, optional): Path where to save the JSON file.
                                   If None, saves to the same directory as the txt file.
    
    Returns:
        str: Path to the saved JSON file, or None if failed.
    """
    # Read the data
    questions_list = read_questions_txt()
    
    if not questions_list:
        return None
    
    # Determine output path
    if output_path is None:
        txt_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1')
        output_path = os.path.join(txt_dir, 'questions.json')
    
    try:
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(questions_list, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved questions JSON to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving questions JSON file: {str(e)}")
        return None


def read_pdfs_from_folder(folder_path):
    """
    Read all PDF files from a given folder and return their document names without paths.
    
    Args:
        folder_path (str): Path to the folder containing PDF files.
                          Can be absolute or relative path.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - 'filename': The PDF filename without path (string)
              - 'full_path': The complete path to the PDF file (string)
              - 'size': File size in bytes (integer)
    """
    pdf_files = []
    
    try:
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist")
            return []
        
        if not os.path.isdir(folder_path):
            print(f"Error: '{folder_path}' is not a directory")
            return []
        
        # Search for PDF files recursively in the folder
        # Using glob to find all PDF files (case-insensitive)
        pdf_pattern = os.path.join(folder_path, "**", "*.pdf")
        pdf_files_found = glob.glob(pdf_pattern, recursive=True)
        
        # Also search for uppercase PDF extension
        pdf_pattern_upper = os.path.join(folder_path, "**", "*.PDF")
        pdf_files_found.extend(glob.glob(pdf_pattern_upper, recursive=True))
        
        # Remove duplicates (in case both .pdf and .PDF exist)
        pdf_files_found = list(set(pdf_files_found))
        
        # Process each PDF file
        for pdf_path in pdf_files_found:
            try:
                # Get filename without path
                filename = os.path.basename(pdf_path)
                
                # Get file size
                file_size = os.path.getsize(pdf_path)
                
                # Create dictionary entry
                pdf_info = {
                    'filename': filename,
                    'full_path': pdf_path,
                    'size': file_size
                }
                
                pdf_files.append(pdf_info)
                
            except OSError as e:
                print(f"Warning: Could not access file '{pdf_path}': {str(e)}")
                continue
        
        # Sort by filename for consistent ordering
        pdf_files.sort(key=lambda x: x['filename'])
        
        return pdf_files
        
    except Exception as e:
        print(f"Error reading PDFs from folder '{folder_path}': {str(e)}")
        return []


def get_pdf_filenames_only(folder_path):
    """
    Get only the filenames (without paths) of all PDF files in a folder.
    
    Args:
        folder_path (str): Path to the folder containing PDF files.
    
    Returns:
        list: A list of PDF filenames (strings) without paths.
    """
    pdf_files = read_pdfs_from_folder(folder_path)
    return [pdf_info['filename'] for pdf_info in pdf_files]


def save_pdf_list_json(folder_path, output_path=None):
    """
    Read all PDFs from a folder and save the list as a JSON file.
    
    Args:
        folder_path (str): Path to the folder containing PDF files.
        output_path (str, optional): Path where to save the JSON file.
                                   If None, saves in the same folder as the PDFs.
    
    Returns:
        str: Path to the saved JSON file, or None if failed.
    """
    # Read the PDF files
    pdf_files = read_pdfs_from_folder(folder_path)
    
    if not pdf_files:
        return None
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(folder_path, 'pdf_files_list.json')
    
    try:
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(pdf_files, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved PDF list JSON to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving PDF list JSON file: {str(e)}")
        return None


def check_pdf_files_in_master_clauses(pdf_folder_path=None, master_clauses_json_path=None):
    """
    Check if PDF filenames from a folder exist in the master_clauses.json file.
    
    Args:
        pdf_folder_path (str, optional): Path to the folder containing PDF files.
                                       If None, uses the default CUAD_v1 full_contract_pdf folder.
        master_clauses_json_path (str, optional): Path to the master_clauses.json file.
                                                If None, uses the default path.
    
    Returns:
        dict: A dictionary containing:
              - 'total_pdfs': Total number of PDF files found
              - 'total_master_clauses': Total number of entries in master_clauses.json
              - 'matched_files': List of PDF files that exist in master_clauses.json
              - 'unmatched_files': List of PDF files that don't exist in master_clauses.json
              - 'missing_pdfs': List of master_clauses entries that don't have corresponding PDF files
    """
    # Set default paths if not provided
    if pdf_folder_path is None:
        pdf_folder_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1', 'full_contract_pdf')
    
    if master_clauses_json_path is None:
        master_clauses_json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1', 'master_clauses.json')
    
    try:
        # Read PDF files from folder
        print(f"Reading PDF files from: {pdf_folder_path}")
        pdf_files = read_pdfs_from_folder(pdf_folder_path)
        pdf_filenames = {pdf_info['filename'] for pdf_info in pdf_files}
        
        # Read master_clauses.json
        print(f"Reading master_clauses.json from: {master_clauses_json_path}")
        with open(master_clauses_json_path, 'r', encoding='utf-8') as jsonfile:
            master_clauses = json.load(jsonfile)
        
        # Extract filenames from master_clauses.json
        master_clauses_filenames = {entry['Filename'] for entry in master_clauses}
        
        # Find matches and mismatches
        matched_files = list(pdf_filenames.intersection(master_clauses_filenames))
        unmatched_files = list(pdf_filenames - master_clauses_filenames)
        missing_pdfs = list(master_clauses_filenames - pdf_filenames)
        
        # Sort lists for consistent output
        matched_files.sort()
        unmatched_files.sort()
        missing_pdfs.sort()
        
        result = {
            'total_pdfs': len(pdf_files),
            'total_master_clauses': len(master_clauses),
            'matched_files': matched_files,
            'unmatched_files': unmatched_files,
            'missing_pdfs': missing_pdfs
        }
        
        # Print summary
        print(f"\n=== PDF Files vs Master Clauses Comparison ===")
        print(f"Total PDF files found: {result['total_pdfs']}")
        print(f"Total master clauses entries: {result['total_master_clauses']}")
        print(f"Matched files: {len(result['matched_files'])}")
        print(f"Unmatched PDF files: {len(result['unmatched_files'])}")
        print(f"Missing PDF files: {len(result['missing_pdfs'])}")
        
        if result['unmatched_files']:
            print(f"\nUnmatched PDF files (first 10):")
            for i, filename in enumerate(result['unmatched_files'][:10]):
                print(f"  {i+1}. {filename}")
        
        if result['missing_pdfs']:
            print(f"\nMissing PDF files (first 10):")
            for i, filename in enumerate(result['missing_pdfs'][:10]):
                print(f"  {i+1}. {filename}")
        
        return result
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {str(e)}")
        return None
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        return None


def save_comparison_report(comparison_result, output_path=None):
    """
    Save the PDF vs Master Clauses comparison result to a JSON file.
    
    Args:
        comparison_result (dict): Result from check_pdf_files_in_master_clauses()
        output_path (str, optional): Path where to save the report.
                                   If None, saves in the data/CUAD_v1 folder.
    
    Returns:
        str: Path to the saved report file, or None if failed.
    """
    if comparison_result is None:
        return None
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1', 'pdf_master_clauses_comparison.json')
    
    try:
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(comparison_result, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved comparison report to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving comparison report: {str(e)}")
        return None


def get_ans(doc_name, question, master_clauses):
    """
    Get the answer and provenance for a specific document and question from master_clauses data.
    
    Args:
        doc_name (str): The document name/filename to search for
        question (str): The question category to get the answer for
        master_clauses (list): List of dictionaries containing master clauses data
    
    Returns:
        dict: A dictionary containing:
              - 'answer': The answer for the question (from "question-Answer" field)
              - 'provenance': The provenance text (from "question" field)
              - 'found': Boolean indicating if the document and question were found
              - 'doc_name': The document name that was searched for
              - 'question': The question that was searched for
    """
    # Normalize the document name for comparison (remove path if present)
    doc_name_normalized = os.path.basename(doc_name)
    
    # Search for the document in master_clauses
    matching_entry = None
    for entry in master_clauses:
        if entry.get('Filename') == doc_name_normalized:
            matching_entry = entry
            break
    
    if not matching_entry:
        return {
            'answer': None,
            'provenance': None,
            'found': False,
            'doc_name': doc_name_normalized,
            'question': question,
            'error': f"Document '{doc_name_normalized}' not found in master_clauses"
        }
    
    # Construct the field names for answer and provenance
    answer_field = f"{question}-Answer"
    provenance_field = question
    
    # Extract answer and provenance
    answer = matching_entry.get(answer_field, None)
    provenance = matching_entry.get(provenance_field, None)
    
    # Check if the question exists in this document
    question_exists = answer_field in matching_entry or provenance_field in matching_entry
    
    return {
        'answer': answer,
        'provenance': provenance,
        'found': question_exists,
        'doc_name': doc_name_normalized,
        'question': question,
        'answer_field': answer_field,
        'provenance_field': provenance_field
    }


def get_all_answers_for_document(doc_name, master_clauses, questions_data=None):
    """
    Get all answers and provenance for a specific document from master_clauses data.
    
    Args:
        doc_name (str): The document name/filename to search for
        master_clauses (list): List of dictionaries containing master clauses data
        questions_data (list, optional): List of question dictionaries. If None, uses default questions.
    
    Returns:
        dict: A dictionary containing:
              - 'doc_name': The document name
              - 'total_questions': Total number of questions processed
              - 'found_questions': Number of questions with answers
              - 'answers': Dictionary with question categories as keys and answer/provenance as values
    """
    # If no questions_data provided, use default questions from the questions.json
    if questions_data is None:
        questions_json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1', 'questions.json')
        try:
            with open(questions_json_path, 'r', encoding='utf-8') as jsonfile:
                questions_data = json.load(jsonfile)
        except FileNotFoundError:
            print(f"Warning: Could not find questions.json at {questions_json_path}")
            return None
    
    # Normalize the document name
    doc_name_normalized = os.path.basename(doc_name)
    
    # Search for the document in master_clauses
    matching_entry = None
    for entry in master_clauses:
        if entry.get('Filename') == doc_name_normalized:
            matching_entry = entry
            break
    
    if not matching_entry:
        return {
            'doc_name': doc_name_normalized,
            'total_questions': len(questions_data),
            'found_questions': 0,
            'answers': {},
            'error': f"Document '{doc_name_normalized}' not found in master_clauses"
        }
    
    # Extract all answers for this document
    answers = {}
    found_count = 0
    
    for question_info in questions_data:
        question_category = question_info.get('category', '')
        answer_field = f"{question_category}-Answer"
        provenance_field = question_category
        
        answer = matching_entry.get(answer_field, None)
        provenance = matching_entry.get(provenance_field, None)
        
        if answer is not None or provenance is not None:
            found_count += 1
        
        answers[question_category] = {
            'answer': answer,
            'provenance': provenance,
            'answer_format': question_info.get('answer_format', ''),
            'group': question_info.get('group', '')
        }
    
    return {
        'doc_name': doc_name_normalized,
        'total_questions': len(questions_data),
        'found_questions': found_count,
        'answers': answers
    }


def save_document_answers_json(doc_name, master_clauses, questions_data=None, output_path=None):
    """
    Get all answers for a document and save them to a JSON file.
    
    Args:
        doc_name (str): The document name/filename to search for
        master_clauses (list): List of dictionaries containing master clauses data
        questions_data (list, optional): List of question dictionaries
        output_path (str, optional): Path where to save the JSON file
    
    Returns:
        str: Path to the saved JSON file, or None if failed
    """
    # Get all answers for the document
    document_answers = get_all_answers_for_document(doc_name, master_clauses, questions_data)
    
    if document_answers is None:
        return None
    
    # Determine output path
    if output_path is None:
        doc_name_clean = os.path.splitext(os.path.basename(doc_name))[0]
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1', f'{doc_name_clean}_answers.json')
    
    try:
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(document_answers, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved document answers to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving document answers: {str(e)}")
        return None


if __name__ == "__main__":
    # # Example usage for master_clauses.csv
    # print("=== Master Clauses CSV ===")
    # data = read_master_clauses_csv()
    # if data:
    #     print(f"First row keys: {list(data[0].keys())}")
    #     print(f"Total columns: {len(data[0].keys())}")
    #     print(f"Sample data from first row:")
    #     for key, value in list(data[0].items())[:5]:  # Show first 5 columns
    #         print(f"  {key}: {value}")
    
    # # Save master_clauses as JSON
    # json_path = save_master_clauses_json()
    # if json_path:
    #     print(f"Master clauses JSON file saved at: {json_path}")
    
    # print("\n=== Questions TXT ===")
    # # Example usage for questions.txt
    # questions = read_questions_txt()
    # if questions:
    #     print(f"Total questions: {len(questions)}")
    #     print(f"Sample question (first one):")
    #     sample_q = questions[0]
    #     for key, value in sample_q.items():
    #         print(f"  {key}: {value}")
        
    #     print(f"\nSample question (last one):")
    #     sample_q = questions[-1]
    #     for key, value in sample_q.items():
    #         print(f"  {key}: {value}")
    
    # # Save questions as JSON
    # questions_json_path = save_questions_json()
    # if questions_json_path:
    #     print(f"Questions JSON file saved at: {questions_json_path}")
    
    print("\n=== PDF Files from Folder ===")
    # Example usage for reading PDFs from folder
    pdf_folder = os.path.join(os.path.dirname(__file__), '..', 'data', 'CUAD_v1', 'full_contract_pdf')
    
    # # Test reading PDFs from folder
    # pdf_files = read_pdfs_from_folder(pdf_folder)
    # if pdf_files:
    #     print(f"Found {len(pdf_files)} PDF files")
    #     print("Sample PDF files (first 5):")
    #     for i, pdf_info in enumerate(pdf_files[:5]):
    #         print(f"  {i+1}. {pdf_info['filename']} ({pdf_info['size']} bytes)")
        
    #     # Test getting only filenames
    #     print(f"\nFirst 10 PDF filenames only:")
    #     filenames = get_pdf_filenames_only(pdf_folder)
    #     for i, filename in enumerate(filenames[:10]):
    #         print(f"  {i+1}. {filename}")
    
    
    # print("\n=== PDF Files vs Master Clauses Comparison ===")
    # # Compare PDF files with master_clauses.json
    # comparison_result = check_pdf_files_in_master_clauses()

    for pdfs in pdf_files:
        print(pdfs['filename'])
    
    print("\n=== Testing get_ans Function ===")
    # Test the get_ans function
    master_clauses_data = read_master_clauses_csv()
    if master_clauses_data:
        
        
        result = get_ans(sample_doc, sample_question, master_clauses_data)
        
        
        # # Test getting all answers for a document
        # print(f"\n=== Testing get_all_answers_for_document ===")
        # all_answers = get_all_answers_for_document(sample_doc, master_clauses_data)
        # if all_answers:
        #     print(f"Document: {all_answers['doc_name']}")
        #     print(f"Total questions: {all_answers['total_questions']}")
        #     print(f"Found answers: {all_answers['found_questions']}")
            
        #     # Show a few sample answers with provenance
        #     print(f"\nSample answers with provenance:")
        #     for i, (question, answer_data) in enumerate(list(all_answers['answers'].items())[:5]):
        #         if answer_data['answer'] is not None or answer_data['provenance'] is not None:
        #             print(f"  {i+1}. {question}:")
        #             print(f"     Answer: {answer_data['answer']}")
        #             print(f"     Provenance: {answer_data['provenance']}")
        #             print()
    
