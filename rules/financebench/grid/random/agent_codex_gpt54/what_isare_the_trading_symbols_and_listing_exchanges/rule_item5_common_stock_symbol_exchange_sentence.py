def rule_item5_common_stock_symbol_exchange_sentence(doc: dict) -> list[dict]:
    """Match Item 5 or Common Stock sentences that state the symbol and exchange for common stock."""
    try:
        import re

        path_re = re.compile(r"item 5|market for .*common (?:equity|stock)|common stock", re.IGNORECASE)
        exchange_re = re.compile(r"new york stock exchange|nyse|nasdaq(?: global select market| global market| capital market)?", re.IGNORECASE)
        symbol_phrase_re = re.compile(r"under the symbol|trades under the symbol|symbol [\"“]?[A-Z]{1,6}", re.IGNORECASE)
        listing_re_a = re.compile(r"(?:common stock|ordinary shares?).{0,160}(?:listed|traded|quoted)", re.IGNORECASE)
        listing_re_b = re.compile(r"principal market.{0,80}(?:common stock|ordinary shares?)", re.IGNORECASE)

        def normalize(text: str) -> str:
            return " ".join((text or "").split()).strip()

        return [
            span for span in doc.get("texts", [])
            if span.get("label") != "table"
            and path_re.search(normalize(((span.get("structure") or {}).get("path_text")) or ""))
            and (
                listing_re_a.search(normalize(span.get("text") or ""))
                or listing_re_b.search(normalize(span.get("text") or ""))
            )
            and exchange_re.search(normalize(span.get("text") or ""))
            and symbol_phrase_re.search(normalize(span.get("text") or ""))
        ]
    except Exception:
        return []
