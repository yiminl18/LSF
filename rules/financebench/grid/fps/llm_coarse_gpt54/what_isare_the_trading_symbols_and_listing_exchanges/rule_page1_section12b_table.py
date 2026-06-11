def rule_page1_section12b_table(doc: dict) -> list[dict]:
    """Match page-1 tables/spans around 'Securities registered pursuant to Section 12(b) of the Act'."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r"Section\s+12\(b\)", txt, re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 8)):
                    out.append(texts[j])
        return out
    except Exception:
        return []
