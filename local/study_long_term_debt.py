"""Study documents to find long-term debt spans and their structural patterns."""

import json
import re
from pathlib import Path

# Ground truth for long-term debt question
GROUND_TRUTH = {
    "AMCOR_2019_10K": "5,314.4 million",
    "COSTCO_2017_10K": "6,573",
    "BOEING_2018_10K": "$10,657 million",
    "AMAZON_2018_10K": "$50,708 million",
    "EBAY_2021_10K": "$7,727 million",
    "AMAZON_2016_10K": "$7,694 million",
    "CORNING_2022_10K": "6,687 million",
    "NIKE_2021_10K": "$9,413 million",
    "LOCKHEEDMARTIN_2022_10K": "$15,547 million",
    "JOHNSON_JOHNSON_2022_10K": "$26.9 billion",
}

def normalize_number(s):
    """Extract just the number part for matching."""
    # Remove $ and convert to lowercase
    s = s.replace("$", "").strip().lower()
    # Remove "million", "billion" suffixes
    s = re.sub(r'\s*(million|billion)s?\s*', '', s)
    return s.strip()

def load_doc(doc_name):
    """Load a document JSON."""
    path = Path(f"data/financebench/processing/{doc_name}_reconstructed.json")
    with open(path) as f:
        return json.load(f)

def find_answer_spans(doc, answer):
    """Find spans containing the answer."""
    answer_num = normalize_number(answer)
    matching_spans = []

    for i, span in enumerate(doc.get("texts", [])):
        text = span.get("text", "")
        text_lower = text.lower()

        # Check if the number appears in the text
        if answer_num in text_lower or answer_num.replace(",", "") in text_lower.replace(",", ""):
            matching_spans.append((i, span))

        # Also check table cells
        if span.get("label") == "table" and span.get("table_data"):
            cells = span.get("table_data", {}).get("cells", [])
            for cell in cells:
                cell_text = cell.get("text", "").lower()
                if answer_num in cell_text or answer_num.replace(",", "") in cell_text.replace(",", ""):
                    if (i, span) not in matching_spans:
                        matching_spans.append((i, span))
                    break

    return matching_spans

def analyze_span(span):
    """Extract structural info from a span."""
    return {
        "page_no": span.get("page_no"),
        "label": span.get("label"),
        "bold": span.get("bold"),
        "size": span.get("size"),
        "level": span.get("structure", {}).get("level"),
        "path_text": span.get("structure", {}).get("path_text", ""),
        "text_preview": span.get("text", "")[:200],
    }

def main():
    print("Studying long-term debt spans in documents\n")
    print("=" * 80)

    for doc_name, answer in GROUND_TRUTH.items():
        print(f"\n{doc_name}")
        print(f"  Ground truth: {answer}")

        try:
            doc = load_doc(doc_name)
            total_spans = len(doc.get("texts", []))
            print(f"  Total spans: {total_spans}")

            matching = find_answer_spans(doc, answer)
            print(f"  Matching spans: {len(matching)}")

            for idx, span in matching:
                info = analyze_span(span)
                print(f"\n  Span {idx}:")
                print(f"    page_no: {info['page_no']}")
                print(f"    label: {info['label']}")
                print(f"    level: {info['level']}")
                print(f"    path_text: {info['path_text']}")
                print(f"    bold: {info['bold']}")
                print(f"    text_preview: {info['text_preview'][:100]}...")

                # Look for long-term debt keywords nearby
                if "long-term" in info['text_preview'].lower() or "long term" in info['text_preview'].lower():
                    print(f"    ** Contains 'long-term' keyword **")
                if span.get("label") == "table":
                    cells = span.get("table_data", {}).get("cells", [])
                    debt_headers = [c for c in cells if "debt" in c.get("text", "").lower() and c.get("is_row_header")]
                    if debt_headers:
                        print(f"    ** Table has debt row headers: {[c['text'] for c in debt_headers]} **")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("\nSummary of patterns:")

    # Aggregate pattern analysis
    patterns = {"labels": {}, "pages": {}, "path_keywords": {}}
    for doc_name, answer in GROUND_TRUTH.items():
        try:
            doc = load_doc(doc_name)
            matching = find_answer_spans(doc, answer)
            for idx, span in matching:
                label = span.get("label", "unknown")
                page = span.get("page_no", 0)
                path_text = span.get("structure", {}).get("path_text", "").lower()

                patterns["labels"][label] = patterns["labels"].get(label, 0) + 1
                patterns["pages"][page] = patterns["pages"].get(page, 0) + 1

                # Extract path keywords
                for kw in ["item 8", "balance sheet", "financial statements", "debt", "liabilities", "consolidated"]:
                    if kw in path_text:
                        patterns["path_keywords"][kw] = patterns["path_keywords"].get(kw, 0) + 1
        except:
            pass

    print(f"\nLabels: {patterns['labels']}")
    print(f"\nPages: {sorted(patterns['pages'].items())}")
    print(f"\nPath keywords: {patterns['path_keywords']}")

if __name__ == "__main__":
    main()
