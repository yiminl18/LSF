def rule_page1_cover_page_phone_pattern_excluding_market_values(doc: dict) -> list[dict]:
    """Match page-1 phone-like spans while excluding obvious market value/share count lines."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(?:\+\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}")
        bad_re = re.compile(r"aggregate market value|shares outstanding|common stock outstanding|market value", re.I)
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            blob = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).strip()
            if phone_re.search(blob) and not bad_re.search(blob):
                out.append(span)
        return out
    except Exception:
        return []
