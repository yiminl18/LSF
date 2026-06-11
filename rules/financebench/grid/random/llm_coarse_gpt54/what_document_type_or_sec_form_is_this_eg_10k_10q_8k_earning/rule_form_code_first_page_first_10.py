def rule_form_code_first_page_first_10(doc: dict) -> list[dict]:
    """Match form-code spans among the first 10 page-1 spans."""
    import re
    try:
        page1 = [s for s in doc.get("texts", []) if s.get("page_no") == 1][:10]
        return [s for s in page1 if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", s.get("text") or "", re.I)]
    except Exception:
        return []
