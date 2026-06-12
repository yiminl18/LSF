def rule_page12_fiscal_year_period(doc: dict) -> list[dict]:
    """Match page-1/2 annual-report spans containing a fiscal-year end phrase."""
    try:
        import re
        return [
            s for s in doc.get("texts", [])
            if (s.get("page_no") or 0) <= 2
            and s.get("label") in {"text", "section_header", "checkbox_selected"}
            and re.search(
                r"\bfor the fiscal year ended\b|\bfor the year ended\b|\bfiscal year ended\b",
                (s.get("text") or "").lower(),
            )
        ]
    except Exception:
        return []
