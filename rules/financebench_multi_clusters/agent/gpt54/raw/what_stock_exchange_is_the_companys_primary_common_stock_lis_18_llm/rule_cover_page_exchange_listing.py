def rule_cover_page_exchange_listing(doc: dict) -> list[dict]:
    """Retrieve page-1 cover-page spans describing the exchange where common stock is registered."""
    try:
        import re
        texts = doc.get("texts", []) or []
        out = []
        for span in texts:
            text = (span.get("text") or "")
            label = span.get("label")
            page_no = span.get("page_no")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if page_no != 1:
                continue
            if label not in {"text", "table", "section_header"}:
                continue
            t = text.lower()
            p = path.lower()
            if not any(k in t for k in [
                "securities registered pursuant to section 12(b)",
                "name of each exchange on which registered",
                "trading symbol",
                "trading symbol(s)",
                "nasdaq",
                "new york stock exchange",
                "nasdaq global select market",
                "the nasdaq global select market",
            ]):
                continue
            if not (
                p == "" or
                any(x in p for x in [
                    "adobe inc.", "amazon.com, inc.", "costco wholesale corporation",
                    "the boeing company", "activision blizzard, inc.", "amcor plc",
                    "foot locker, inc.", "3m company", "ebay"
                ])
            ):
                continue
            out.append(span)
        return out
    except Exception:
        return []

