def rule_item5_common_stock_exchange_sentence(doc: dict) -> list[dict]:
    """Match Item 5 market-information sentences that say the common stock is listed or traded on an exchange."""
    try:
        import re

        exchange_re = re.compile(
            r"\b(?:new york stock exchange|nyse|nasdaq(?: global select market| global market| capital market)?|the nasdaq global select market|the nasdaq global market|the new york stock exchange)\b",
            re.IGNORECASE,
        )
        path_re = re.compile(r"item 5\.|market for .*common (?:equity|stock)|common stock", re.IGNORECASE)
        sentence_re = re.compile(
            r"\b(?:common stock|ordinary shares?).{0,100}\b(?:listed|traded|quoted)\s+on\b",
            re.IGNORECASE,
        )

        def normalize(text: str) -> str:
            return " ".join((text or "").split()).strip()

        return [
            span for span in doc.get("texts", [])
            if span.get("label") != "table"
            and exchange_re.search(normalize(span.get("text") or ""))
            and sentence_re.search(normalize(span.get("text") or ""))
            and path_re.search(normalize(((span.get("structure") or {}).get("path_text")) or ""))
        ]
    except Exception:
        return []
