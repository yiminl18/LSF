def rule_financial_condition_text_with_debt_as_of_year_end(doc: dict) -> list[dict]:
    """Match text spans containing debt plus as-of year-end phrasing anywhere in the document."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "section_header"}:
                continue
            txt = (span.get("text") or "")
            if re.search(r"\bdebt\b", txt, re.I) and re.search(r"\bas of\b|\bat the end of\b|\bat year[- ]end\b", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
