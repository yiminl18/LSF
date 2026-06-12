def rule_table_after_ffo1_header(doc: dict) -> list[dict]:
    """Match tables immediately following a section header for Table FFO-1 / Summary of Fiscal Operations."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") == "section_header":
                txt = span.get("text") or ""
                if re.search(r'(Table\s*)?FFO[-\s]?1', txt, re.I) or re.search(r'Summary of Fiscal Operations', txt, re.I):
                    for j in range(i+1, min(i+5, len(texts))):
                        if texts[j].get("label") == "table":
                            out.append(texts[j])
        return out
    except Exception:
        return []
