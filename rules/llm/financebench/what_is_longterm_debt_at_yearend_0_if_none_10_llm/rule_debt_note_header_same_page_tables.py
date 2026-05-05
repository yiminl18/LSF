def rule_debt_note_header_same_page_tables(doc: dict) -> list[dict]:
    """Match tables on the same page as a debt-related section header."""
    import re
    try:
        debt_pages = set()
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                txt = (span.get("text") or "").lower()
                if re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt) or "notes payable" in txt:
                    debt_pages.add(span.get("page_no"))
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table" and span.get("page_no") in debt_pages:
                out.append(span)
        return out
    except Exception:
        return []
