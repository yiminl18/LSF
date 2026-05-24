def rule_page1_h1_form(doc: dict) -> list[dict]:
    """Match page-1 H1 section headers that are form names."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            level = ((span.get("structure") or {}).get("level") or "")
            if span.get("page_no") == 1 and level == "H1" and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (span.get("text") or ""), re.I):
                out.append(span)
        return out
    except Exception:
        return []
