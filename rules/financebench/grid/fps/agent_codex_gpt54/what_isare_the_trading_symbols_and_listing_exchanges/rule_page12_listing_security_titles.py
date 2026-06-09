def rule_page12_listing_security_titles(doc: dict) -> list[dict]:
    """Match page-1/2 common-stock and ordinary-share title spans inside the listing block."""
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
        if not has_listing_block:
            return []

        results = []
        for span in texts:
            if span.get("page_no", 999) > 2:
                continue

            lowered = span.get("text", "").lower()
            if (
                "aggregate market value" in lowered
                or "shares outstanding" in lowered
                or "number of shares outstanding" in lowered
            ):
                continue

            stripped = lowered.strip()
            if (
                stripped.startswith("common stock")
                or stripped.startswith("ordinary shares")
                or re.match(r"class\s+[a-z]\s+common stock", stripped)
            ):
                results.append(span)

        return results
    except Exception:
        return []
