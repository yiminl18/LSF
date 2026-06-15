def rule_page1_3_bracketed_exchange_ticker_sentence(doc: dict) -> list[dict]:
    """Match page 1-3 release sentences with bracketed or parenthesized exchange:ticker pairs."""
    try:
        import re

        pattern = re.compile(
            r"[\[(](?:nyse|nasdaq(?:\s+global\s+select\s+market)?)\s*:\s*[A-Z0-9./%-]{1,12}[\])]",
            re.I,
        )
        return [
            s for s in doc.get("texts", [])
            if s.get("page_no", 999) <= 3
            and pattern.search(s.get("text", ""))
        ]
    except Exception:
        return []
