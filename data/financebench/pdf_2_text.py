import sys
import os
import fitz  # pymupdf


def pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def process_all(raw_dir: str, text_dir: str):
    pdfs = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    total = len(pdfs)
    for i, pdf_name in enumerate(pdfs, 1):
        out_name = pdf_name.replace(".pdf", ".txt")
        out_path = os.path.join(text_dir, out_name)
        if os.path.exists(out_path):
            print(f"[{i}/{total}] Skipping (exists): {out_name}")
            continue
        pdf_path = os.path.join(raw_dir, pdf_name)
        try:
            text = pdf_to_text(pdf_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[{i}/{total}] Done: {out_name}")
        except Exception as e:
            print(f"[{i}/{total}] ERROR {pdf_name}: {e}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) == 2:
        # Single PDF mode
        pdf_path = sys.argv[1]
        text = pdf_to_text(pdf_path)
        print(text)
    else:
        # Batch mode: process all PDFs in raw/
        raw_dir = os.path.join(base, "raw")
        text_dir = os.path.join(base, "text")
        os.makedirs(text_dir, exist_ok=True)
        process_all(raw_dir, text_dir)
