def rule_first_financial_table_after_item8(doc: dict) -> list[dict]:
    """Match the first table after an Item 8 header, which is often a primary financial statement."""
    import re
    try:
        texts = doc.get("texts", [])
        for i, s in enumerate(texts):
            if s.get("label") == "section_header" and re.search(r"item\s*8", s.get("text", "") or "", re.I):
                for j in range(i + 1, min(i + 20, len(texts))):
                    if texts[j].get("label") == "table":
                        return [texts[j]]
        return []
    except Exception:
        return []
