#!/usr/bin/env python3
"""Analyze telephone number patterns across sampled documents."""

import json
import re
from pathlib import Path

# Sampled documents and their ground truth telephone numbers
SAMPLED_DOCS = {
    "AMCOR_2019_10K": "+44 117 9753200",
    "COSTCO_2017_10K": "(425) 313-8100",
    "BOEING_2018_10K": "(312) 544-2000",
    "AMAZON_2018_10K": "(206) 266-1000",
    "EBAY_2021_10K": "(408) 376-7108",
    "AMAZON_2016_10K": "(206) 266-1000",
    "CORNING_2022_10K": "607-974-9000",
    "NIKE_2021_10K": "(503) 671-6453",
    "LOCKHEEDMARTIN_2022_10K": "(301) 897-6000",
    "JOHNSON_JOHNSON_2022_10K": "(732) 524-0400",
}

DATA_DIR = Path("/home/yiminglin/LSF/data/financebench/processing")

def load_doc(doc_name):
    path = DATA_DIR / f"{doc_name}_reconstructed.json"
    with open(path) as f:
        return json.load(f)

def find_phone_spans(doc, phone):
    """Find spans containing the phone number."""
    results = []
    texts = doc.get("texts", [])
    for i, span in enumerate(texts):
        text = span.get("text", "")
        # Normalize phone for matching
        phone_norm = re.sub(r'\s+', '', phone)
        text_norm = re.sub(r'\s+', '', text)
        if phone_norm in text_norm or phone in text:
            results.append((i, span))
    return results

def main():
    print("=" * 80)
    print("TELEPHONE NUMBER PATTERN ANALYSIS")
    print("=" * 80)

    for doc_name, phone in SAMPLED_DOCS.items():
        print(f"\n{'='*60}")
        print(f"Document: {doc_name}")
        print(f"Ground truth phone: {phone}")
        print("="*60)

        doc = load_doc(doc_name)
        matches = find_phone_spans(doc, phone)

        if not matches:
            print("WARNING: No spans found containing the phone number!")
            # Try searching for keywords
            texts = doc.get("texts", [])
            for i, span in enumerate(texts):
                text = span.get("text", "").lower()
                if "telephone" in text:
                    print(f"\n  Span {i} contains 'telephone':")
                    print(f"    Page: {span.get('page_no')}")
                    print(f"    Label: {span.get('label')}")
                    print(f"    Text (first 300 chars): {span.get('text', '')[:300]}")
        else:
            for idx, span in matches:
                print(f"\nSpan index: {idx}")
                print(f"  Page: {span.get('page_no')}")
                print(f"  Label: {span.get('label')}")
                print(f"  Bold: {span.get('bold')}")
                print(f"  Size: {span.get('size')}")
                print(f"  Level: {span.get('structure', {}).get('level')}")
                print(f"  Path text: {span.get('structure', {}).get('path_text', '')[:100]}")
                text = span.get('text', '')
                print(f"  Text length: {len(text)}")
                # Show context around phone number
                phone_idx = text.find(phone)
                if phone_idx == -1:
                    # Try without spaces
                    phone_norm = re.sub(r'\s+', '', phone)
                    text_norm = re.sub(r'\s+', '', text)
                    phone_idx = text_norm.find(phone_norm)
                    if phone_idx >= 0:
                        print(f"  Phone location: normalized match at ~{phone_idx}")
                else:
                    start = max(0, phone_idx - 50)
                    end = min(len(text), phone_idx + len(phone) + 50)
                    context = text[start:end]
                    print(f"  Context around phone: ...{context}...")

if __name__ == "__main__":
    main()
