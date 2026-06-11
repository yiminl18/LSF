def rule_toc_item_15_exhibits_table(doc: dict) -> list[dict]:
    """Match table-of-contents tables containing Item 15 Exhibits, which often point to the answer location."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            if re.search(r"item\s*15", txt, re.I) and re.search(r"exhibits?", txt, re.I):
                out.append(span)
    except Exception:
        return []
    return out
