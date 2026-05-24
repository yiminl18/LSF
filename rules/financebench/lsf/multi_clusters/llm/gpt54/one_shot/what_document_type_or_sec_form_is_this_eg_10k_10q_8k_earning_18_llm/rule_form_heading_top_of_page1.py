def rule_form_heading_top_of_page1(doc: dict) -> list[dict]:
    """Match SEC form headings on page 1 among the first 25 page-1 spans."""
    import re
    try:
        page1 = [s for s in doc.get("texts", []) if s.get("page_no") == 1][:25]
        return [s for s in page1 if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", (s.get("text") or "").strip(), re.I)]
    except Exception:
        return []
