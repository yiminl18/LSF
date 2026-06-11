def rule_page1_section_header_with_state_value(doc: dict) -> list[dict]:
    """Match page-1 section headers whose text is a likely jurisdiction value."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                if re.fullmatch(r"Delaware|Washington|New York|Jersey(?: \(Channel Islands\))?", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
