def rule_form_heading_all_caps(doc: dict) -> list[dict]:
    """Match all-caps page-1 section headers that are exact form names."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and re.fullmatch(r"FORM\s+(10-K|10-Q|8-K)", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
