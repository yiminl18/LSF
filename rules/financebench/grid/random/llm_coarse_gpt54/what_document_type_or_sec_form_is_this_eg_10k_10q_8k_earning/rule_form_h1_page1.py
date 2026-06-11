def rule_form_h1_page1(doc: dict) -> list[dict]:
    """Match page-1 H1/section_header spans whose text starts with FORM and contains a common SEC form code."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
