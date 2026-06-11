def rule_page1_bold_address_line_without_label(doc: dict) -> list[dict]:
    """Match bold page-1 spans that look like address lines near company-identification content."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            if re.search(r'\b\d+\s+\S+', txt) and (
                re.search(r'\b[A-Z]{2}\b', txt) or
                re.search(r'California|Washington|Minnesota|New York|Bristol|United Kingdom', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
