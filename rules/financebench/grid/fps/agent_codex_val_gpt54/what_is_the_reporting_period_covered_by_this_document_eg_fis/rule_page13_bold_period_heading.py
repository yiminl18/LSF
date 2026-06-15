def rule_page13_bold_period_heading(doc: dict) -> list[dict]:
    """Match bold cover or repeated heading spans with an ended-period phrase."""
    try:
        import re
        out = []
        news_release_tables = []
        for s in doc.get("texts", []):
            if (s.get("page_no") or 0) > 3:
                continue
            if s.get("label") not in {"text", "section_header", "checkbox_selected", "list_item", "table"}:
                continue

            text = (s.get("text") or "").lower()
            path = (((s.get("structure") or {}).get("path_text")) or "").lower()
            if not re.search(
                r"\bfor the fiscal year ended\b|\bfor the year ended\b|\bfor the quarterly period ended\b|\bfor the quarter ended\b|\bquarter ended\b|\bquarters ended\b",
                text,
            ):
                continue

            if "news release" in path and s.get("label") == "table":
                news_release_tables.append(s)
                continue

            if (
                s.get("bold") == 1
                or "form 10-" in path
                or "current report" in path
            ):
                out.append(s)

        return news_release_tables[:2] if news_release_tables else out
    except Exception:
        return []
