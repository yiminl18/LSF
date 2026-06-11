def rule_page1_text_spans_with_symbol_and_exchange(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning both a ticker-like token and an exchange name."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).strip()
            if span.get("page_no") != 1:
                continue
            if re.search(r"\b[A-Z]{1,5}(?:[/-][A-Z0-9]{1,5})?\d{0,2}\b", txt) and re.search(r"nasdaq|stock exchange|nyse", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
