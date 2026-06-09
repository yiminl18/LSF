def rule_page1_exchange_under_trading_symbol_path(doc: dict) -> list[dict]:
    """Match page-1 exchange values whose breadcrumb is the Trading Symbol(s) cover block."""
    try:
        import re

        exchange_re = re.compile(
            r"\b(?:new york stock exchange|nyse|nasdaq(?: global select market| global market| capital market)?|the nasdaq global select market|the nasdaq global market|the new york stock exchange)\b",
            re.IGNORECASE,
        )
        blocked_re = re.compile(
            r"aggregate market value|holders of record|also traded on|also listed on|closing sale price|closing price|stockholders of record|the number of shares",
            re.IGNORECASE,
        )

        def normalize(text: str) -> str:
            return " ".join((text or "").split()).strip()

        return [
            span for span in doc.get("texts", [])
            if span.get("page_no") == 1
            and span.get("label") != "table"
            and "trading symbol" in normalize(((span.get("structure") or {}).get("path_text")) or "").lower()
            and len(normalize(span.get("text") or "")) <= 120
            and exchange_re.search(normalize(span.get("text") or ""))
            and not blocked_re.search(normalize(span.get("text") or ""))
        ]
    except Exception:
        return []
