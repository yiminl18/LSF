def rule_page13_bold_period_heading(doc: dict) -> list[dict]:
    """Match bold cover or repeated heading spans with an ended-period phrase."""
    try:
        import re
        return [
            s for s in doc.get("texts", [])
            if (s.get("page_no") or 0) <= 3
            and s.get("label") in {"text", "section_header", "checkbox_selected"}
            and re.search(
                r"\bfor the fiscal year ended\b|\bfor the year ended\b|\bfor the quarterly period ended\b|\bfor the quarter ended\b",
                (s.get("text") or "").lower(),
            )
            and (
                s.get("bold") == 1
                or "form 10-" in (((s.get("structure") or {}).get("path_text")) or "").lower()
                or "current report" in (((s.get("structure") or {}).get("path_text")) or "").lower()
            )
        ]
    except Exception:
        return []
