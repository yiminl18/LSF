import os
from pathlib import Path

def count_files_in_folder(folder_path):
    """
    Count the number of files in a folder and all its nested subfolders.
    
    Args:
        folder_path (str): Path to the folder to count files in
        
    Returns:
        int: Total number of files found
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"'{folder_path}' is not a directory")
    
    file_count = 0
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        file_count += len(files)
    
    return file_count

# Alternative implementation using pathlib (more modern approach)
def count_files_in_folder_pathlib(folder_path):
    """
    Count the number of files in a folder and all its nested subfolders using pathlib.
    
    Args:
        folder_path (str): Path to the folder to count files in
        
    Returns:
        int: Total number of files found
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a directory")
    
    # Use rglob to recursively find all files
    file_count = len(list(folder.rglob('*')))
    
    return file_count

# Advanced version with file type filtering
def count_files_in_folder_with_filter(folder_path, file_extension=None):
    """
    Count files in a folder with optional file extension filtering.
    
    Args:
        folder_path (str): Path to the folder to count files in
        file_extension (str, optional): File extension to filter by (e.g., '.pdf', '.txt')
        
    Returns:
        int: Total number of files found
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"'{folder_path}' is not a directory")
    
    file_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        if file_extension:
            # Filter files by extension
            filtered_files = [f for f in files if f.lower().endswith(file_extension.lower())]
            file_count += len(filtered_files)
        else:
            file_count += len(files)
    
    return file_count


import json

def print_first_json_object(json_path: str):
    """
    Print the first JSON object in a JSON file with truncated content (>30 chars).
    Shows the structure of the JSON clearly.

    Args:
        json_path (str): Path to the JSON file.
    """
    def truncate_content(obj):
        """Recursively truncate string content longer than 30 characters."""
        if isinstance(obj, dict):
            return {k: truncate_content(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [truncate_content(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > 300:
            return obj[:300] + "..."
        else:
            return obj

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Truncate the content
    truncated_data = truncate_content(data)

    # If the file contains a list of objects
    if isinstance(truncated_data, list) and truncated_data:
        print(json.dumps(truncated_data[0], indent=2, ensure_ascii=False))
    # If the file itself is just a single object
    elif isinstance(truncated_data, dict):
        print(json.dumps(truncated_data, indent=2, ensure_ascii=False))
    else:
        print("No valid JSON object found.")



# Example usage and testing
if __name__ == "__main__":
    print_first_json_object('/Users/yiminglin/Documents/Codebase/LSF/data/CUAD_v1/CUAD_v1.json')
