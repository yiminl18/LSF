def rule_page_one_debt_summary_table(doc: dict) -> list[dict]:
    """Match page-1 tables/spans with debt labels, capturing filings where the answer appears in an early summary table."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "") or ""
            if re.search(r"\blong[\-\s]?term debt\b|\bdebt\b|\bborrowings\b", text, re.I):
                if span.get("label") in {"table", "text", "section_header"}:
                    out.append(span)
        return out
    except Exception:
        return []
