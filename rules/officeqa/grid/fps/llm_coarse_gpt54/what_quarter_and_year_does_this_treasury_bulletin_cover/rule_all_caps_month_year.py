def rule_all_caps_month_year(doc: dict) -> list[dict]:
    """Match all-caps month/year spans, common in later bulletins."""
    import re
    try:
        texts = doc.get("texts", [])
        pat = re.compile(
            r"^(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{4}$|"
            r"^(SUMMER|WINTER|SPRING|FALL)\s+ISSUE(?:,\s*|\s+)(SEPTEMBER|DECEMBER|MARCH|JUNE)?\s*\d{4}$|"
            r"^(FIRST|SECOND|THIRD|FOURTH)\s+QUARTER,?\s+FISCAL\s+\d{4}$",
            re.I,
        )
        return [s for s in texts if pat.search((s.get("text") or "").strip())]
    except Exception:
        return []
