def rule_table_after_ffo2_header(doc: dict) -> list[dict]:
    """Match tables immediately following a section header for Table FFO-2 / Budget Receipts by Source."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") == "section_header":
                txt = span.get("text") or ""
                if re.search(r'(Table\s*)?FFO[-\s]?2', txt, re.I) or re.search(r'Budget Receipts by Source', txt, re.I):
                    for j in range(i+1, min(i+5, len(texts))):
                        if texts[j].get("label") == "table":
                            out.append(texts[j])
        return out
    except Exception:
        return []
