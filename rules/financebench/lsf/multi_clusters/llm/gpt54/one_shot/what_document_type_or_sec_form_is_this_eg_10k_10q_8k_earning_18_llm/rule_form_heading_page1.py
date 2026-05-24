def rule_form_heading_page1(doc: dict) -> list[dict]:
    """Match page-1 spans whose text is an SEC form heading like FORM 10-K/10-Q/8-K."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
