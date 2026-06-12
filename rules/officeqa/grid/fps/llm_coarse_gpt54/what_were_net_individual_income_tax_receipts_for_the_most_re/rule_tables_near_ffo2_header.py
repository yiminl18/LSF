def rule_tables_near_ffo2_header(doc: dict) -> list[dict]:
    """Match tables occurring shortly after a section header for FFO-2 / Budget Receipts by Source."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") == "section_header":
                txt = (span.get("text") or "")
                if re.search(r'FFO-2', txt, re.I) or re.search(r'Budget Receipts by Source', txt, re.I):
                    for j in range(i + 1, min(i + 8, len(texts))):
                        s2 = texts[j]
                        if s2.get("label") == "table":
                            out.append(s2)
    except Exception:
        return []
    return out
