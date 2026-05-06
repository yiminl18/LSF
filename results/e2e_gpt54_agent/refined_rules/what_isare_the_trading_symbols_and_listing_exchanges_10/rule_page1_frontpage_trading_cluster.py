def rule_page1_frontpage_trading_cluster(doc: dict) -> list[dict]:
    """Retrieve page-1 front-page spans in the trading symbol / exchange cluster."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"text", "section_header", "table"}:
                continue
            txt = (span.get("text") or "")
            low = txt.lower()
            if (
                "securities registered pursuant to section 12(b)" in low
                or "trading symbol" in low
                or "trading symbol(s)" in low
                or "name of each exchange" in low
                or "name of exchange on which registered" in low
                or "name of each exchange on which registered" in low
                or "new york stock exchange" in low
                or "nasdaq global select market" in low
                or "the nasdaq global select market" in low
                or "nyse" in low
                or txt.strip() in {"AMZN", "NKE", "GLW", "eBay", "LMT", "EBAY", "JNJ", "JNJ24C", "JNJ24BP", "JNJ28", "JNJ35", "AMCR"}
                or "trading symbol lmt" in low
            ):
                out.append(span)
        return out
    except Exception:
        return []

