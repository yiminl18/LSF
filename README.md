# OCR Text Extraction Project

This project provides tools for extracting text from PDFs and images using OCR (Optical Character Recognition).

## Setup

### 1. Create Virtual Environment
```bash
python -m venv ocr_env
source ocr_env/bin/activate  # On macOS/Linux
# or
ocr_env\Scripts\activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install pytesseract PyMuPDF Pillow requests jupyter
```

### 3. Install Tesseract OCR (if not already installed)
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

## Usage

### Python Script
```python
from ocr_notebook import extract_text_from_pdf, extract_text_from_image

# Extract text from PDF
text = extract_text_from_pdf("path/to/document.pdf")
# or from URL
text = extract_text_from_pdf("https://arxiv.org/pdf/2408.09869")

# Extract text from image
text = extract_text_from_image("path/to/image.png")
# or from URL
text = extract_text_from_image("https://example.com/image.jpg")
```

### Jupyter Notebook
1. Start Jupyter: `jupyter notebook`
2. Open `ocr.ipynb` or create a new notebook
3. Import the module:
```python
from ocr_notebook import *
```

## Files

- `ocr.py` - Standalone OCR script
- `ocr_notebook.py` - Module for use in notebooks
- `ocr.ipynb` - Jupyter notebook (to be created)
- `ocr_env/` - Virtual environment directory

## Features

- Extract text from PDF files (local or URL)
- Extract text from images using OCR (local or URL)
- Text analysis and statistics
- Support for multiple file formats
- Error handling and logging

## Example Output

```
OCR Text Extraction Tool
==================================================
Extracting text from: https://arxiv.org/pdf/2408.09869
Text Preview (first 1000 characters):
--------------------------------------------------
Docling Technical Report
Version 1.0
Christoph Auer
Maksym Lysak
...

Total characters: 44250

Text Statistics:
--------------------
Total Characters: 44250
Total Words: 6561
Total Lines: 1305
Non Empty Lines: 1279
Average Words Per Line: 5.03
Average Chars Per Word: 5.63
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Make sure Tesseract is installed and in your PATH
2. **Import errors**: Ensure you're using the virtual environment
3. **PDF processing errors**: Check if the PDF is accessible and not corrupted

### Original Issue
The original `docling` library had compatibility issues with newer versions of Pydantic. This solution uses more stable libraries:
- `pytesseract` for OCR
- `PyMuPDF` for PDF processing
- `Pillow` for image handling

JSON note 

- self_ref: id of phrase with type, e.g., text, body, picture, group
- parent: id of its parent node 
- children: list of ids of children of current node 
- label: type of phrase, e.g., text, body, picture, group
- prov: provenance 
  - page_no
  - bbox 
  - charspan (size of current phrase)
- text: phrase content 