def rule_exhibit_index(doc: dict) -> list[dict]:
    """Match exhibit index tables and text for 10-K, 10-Q, and 8-K documents."""
    try:
        import re
        results = []
        is_8k = "_8K" in doc.get("doc_name", "")

        for span in doc.get("texts", []):
            text = span.get("text", "")
            text_lower = text.lower()
            path = span.get("structure", {}).get("path_text", "").lower()
            label = span.get("label", "")
            page = span.get("page_no", 0)

            if is_8k:
                # For 8-K: prioritize tables in Item 9.01, then early text spans
                if "9.01" in path or "financial statements and exhibits" in path:
                    if label == "table":
                        # Tables containing exhibit listings
                        if "exhibit" in text_lower or "description" in text_lower:
                            results.append(span)
                    elif page <= 3:
                        # Early pages: exhibit descriptions
                        if "exhibit" in text_lower or re.match(r'^\d+\.\d*\.?\s*\w', text):
                            results.append(span)
                        elif label == "section_header" and "9.01" in text_lower:
                            results.append(span)
                # Tables containing Item 9.01 info
                elif label == "table" and "9.01" in text_lower and page <= 3:
                    results.append(span)
            else:
                # For 10-K/10-Q: capture exhibit tables
                if label == "table":
                    if any(x in path for x in ["item 15", "item 6", "exhibit"]):
                        if "exhibit" in text_lower:
                            results.append(span)

        return results
    except Exception:
        return []
