def rule_page1_exchange_name_spans(doc: dict) -> list[dict]:
    """Match page-1 spans whose text is an exchange name."""
    import re
    try:
        pats = [
            r"New York Stock Exchange",
            r"The NASDAQ Global Select Market",
            r"The Nasdaq Global Select Market",
            r"The Nasdaq Stock Market LLC",
            r"NASDAQ\b",
            r"NYSE\b",
            r"Australian Securities Exchange",
        ]
        rx = re.compile("|".join(pats), re.I)
        return [s for s in doc.get("texts", []) if s.get("page_no") == 1 and rx.search(s.get("text") or "")]
    except Exception:
        return []
