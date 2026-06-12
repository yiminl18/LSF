def rule_profile_economy_page_17_summary_table(doc: dict) -> list[dict]:
    """Match summary tables on page 17 in 1987-style issues that may contain labor market averages."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if (
                span.get("page_no") == 17
                and span.get("label") in ("text", "table")
                and "Summary" in path
                and re.search(r'(payroll|employment|job growth)', txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
