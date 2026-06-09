def rule_listing_exchange_spans_and_market_prose(doc: dict) -> list[dict]:
    """Match listing-exchange value spans and later market/ticker prose that names both exchange and symbol."""
    try:
        import re

        texts = doc.get("texts", [])
        cover_text = " ".join(
            s.get("text", "").lower()
            for s in texts
            if s.get("page_no", 999) <= 2
        )
        cover_has_table = any(
            s.get("page_no", 999) <= 2
            and s.get("label") == "table"
            and "trading symbol" in s.get("text", "").lower()
            and "exchange on which registered" in s.get("text", "").lower()
            for s in texts
        )
        cover_has_ticker = any(
            (
                re.fullmatch(r"[A-Z][A-Z0-9]{0,7}", (s.get("text", "") or "").strip())
                or (
                    "trading symbol" in (s.get("text", "") or "").lower()
                    and re.search(r"\b[A-Z][A-Z0-9]{1,7}\b", s.get("text", "") or "")
                )
            )
            for s in texts
            if s.get("page_no", 999) <= 2
        )
        has_listing_block = any(
            marker in cover_text
            for marker in (
                "securities registered pursuant to section 12(b)",
                "trading symbol",
                "exchange on which registered",
            )
        )
        exchange_markers = (
            "new york stock exchange",
            "nasdaq global select market",
            "nasdaq stock market llc",
            "the nasdaq stock market llc",
            "the nasdaq global select market",
            "the nasdaq stock market",
            "nasdaq",
        )
        prose_markers = (
            "under the symbol",
            "under the ticker symbol",
            "trades under the symbol",
            "trades under ticker symbol",
        )

        results = []
        for span in texts:
            text = span.get("text", "") or ""
            lowered = text.lower()

            if re.search(r"[\[(](?:nyse|nasdaq)\s*:\s*[A-Z0-9.-]{1,8}[\])]", text, re.IGNORECASE):
                results.append(span)
                continue

            if (not cover_has_table and not cover_has_ticker) and any(marker in lowered for marker in prose_markers) and any(
                exchange in lowered for exchange in exchange_markers
            ):
                results.append(span)
                continue

            if span.get("page_no", 999) > 2:
                continue

            if not has_listing_block:
                continue

            if (
                "aggregate market value" in lowered
                or "closing price" in lowered
                or "exchange act" in lowered
                or "emerging growth company" in lowered
            ):
                continue

            if any(exchange in lowered for exchange in exchange_markers) or lowered.strip() in {"nasdaq", "nyse"}:
                results.append(span)

        return results
    except Exception:
        return []
