def rule_tables_with_parenthetical_year_end_dates(doc: dict) -> list[dict]:
    """Match tables with year-end date style headers and total assets."""
    import re
    try:
        out = []
        date_re = re.compile(r"(december|january|june|may|august|september|october|november|february|march|april)\s+\d{1,2},?\s+\d{4}", re.I)
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            if "total assets" in text.lower() and date_re.search(text):
                out.append(span)
        return out
    except Exception:
        return []
