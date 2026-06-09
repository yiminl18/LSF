def rule_page12_listing_ticker_spans(doc: dict) -> list[dict]:
    """Match page-1/2 ticker spans plus bracketed exchange:ticker mentions in filings and releases."""
    try:
        import re

        texts = doc.get("texts", [])
        cover_text = " ".join(
            s.get("text", "").lower()
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

        stopwords = {
            "YES", "NO", "FORM", "NYSE", "NASDAQ", "N/A",
            "OR", "CIK", "VA", "NJ", "MD", "WA", "DC",
        }
        results = []

        for span in texts:
            text = (span.get("text", "") or "").strip()
            lowered = text.lower()

            if re.search(r"[\[(](?:nyse|nasdaq)\s*:\s*[A-Z0-9.-]{1,8}[\])]", text, re.IGNORECASE):
                results.append(span)
                continue

            if span.get("page_no", 999) > 2 or not text:
                continue

            if "trading symbol" in lowered and re.search(r"\b[A-Z][A-Z0-9]{1,7}\b", text):
                results.append(span)
                continue

            if not has_listing_block or text in stopwords:
                continue

            if re.fullmatch(r"[A-Z][A-Z0-9]{0,7}", text):
                if (
                    span.get("bold") == 1
                    or span.get("label") == "section_header"
                    or float(span.get("size", 0) or 0) >= 7.0
                ):
                    results.append(span)

        return results
    except Exception:
        return []
