def rule_issue_phrase_with_fiscal_year(doc: dict) -> list[dict]:
    """Match spans containing issue/quarter/fiscal phrasing used in quarterly bulletins."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"\b("
            r"(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter,?\s+fiscal\s+\d{4}|"
            r"(spring|summer|fall|winter)\s+issue(?:\s+of\s+(?:the\s+)?(?:first|second|third|fourth)\s+quarter,?\s+fiscal\s+\d{4})?|"
            r"(spring|summer|fall|winter)\s+issue\s+\d{4}|"
            r"(fall|winter|spring|summer)\s+issue\s+december\s+\d{4}"
            r")\b",
            re.I,
        )
        return [s for s in texts if pat.search((s.get("text") or "").strip())]
    except Exception:
        return []
