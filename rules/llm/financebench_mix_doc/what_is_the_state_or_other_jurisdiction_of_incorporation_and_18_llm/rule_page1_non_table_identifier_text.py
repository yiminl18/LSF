def rule_page1_non_table_identifier_text(doc: dict) -> list[dict]:
    """Match non-table page-1 text/section_header spans likely to be identifier fields."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1 or span.get("label") not in {"text", "section_header"}:
                continue
            text = span.get("text", "") or ""
            if (
                re.search(r"\b\d{2}-\d{7}\b", text)
                or re.search(r"state or other jurisdiction", text, re.I)
                or re.search(r"employer identification", text, re.I)
                or re.search(r"\b(Delaware|Washington|New York|Jersey)\b", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
