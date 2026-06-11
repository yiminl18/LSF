def rule_page1_spans_with_nasdaq_or_nyse(doc: dict) -> list[dict]:
    """Match page-1 spans explicitly containing NASDAQ/NYSE/Stock Exchange names."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text", "") or "")
            if span.get("page_no") == 1 and re.search(r"\bNASDAQ\b|\bNYSE\b|Stock Exchange|Nasdaq Global Select Market", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
