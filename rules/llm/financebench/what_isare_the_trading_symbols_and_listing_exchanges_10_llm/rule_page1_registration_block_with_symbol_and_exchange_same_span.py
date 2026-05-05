def rule_page1_registration_block_with_symbol_and_exchange_same_span(doc: dict) -> list[dict]:
    """Match page 1 spans that contain both a ticker-like token and an exchange name in the same text."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"\b[A-Z]{2,6}(?:\d+[A-Z]{0,3})?\b", txt) and re.search(r"nasdaq|new york stock exchange|nyse", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
