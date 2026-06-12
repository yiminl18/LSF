def rule_page12_quarterly_period(doc: dict) -> list[dict]:
    """Match page-1/2 quarterly-report spans containing the quarter-end phrase."""
    try:
        import re
        return [
            s for s in doc.get("texts", [])
            if (s.get("page_no") or 0) <= 2
            and s.get("label") in {"text", "section_header", "checkbox_selected"}
            and re.search(
                r"\bfor the quarterly period ended\b|\bfor the quarter ended\b|\bquarterly period ended\b|\bquarter ended\b",
                (s.get("text") or "").lower(),
            )
        ]
    except Exception:
        return []
