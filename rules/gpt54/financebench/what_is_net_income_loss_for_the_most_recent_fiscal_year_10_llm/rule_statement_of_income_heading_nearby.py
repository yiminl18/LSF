def rule_statement_of_income_heading_nearby(doc: dict) -> list[dict]:
    """Match section headers for Statement of Income / Operations that likely precede the answer table."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "section_header":
                txt = span.get("text", "") or ""
                if re.search(r"(consolidated )?(statement|statements) of (income|operations|earnings)", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
