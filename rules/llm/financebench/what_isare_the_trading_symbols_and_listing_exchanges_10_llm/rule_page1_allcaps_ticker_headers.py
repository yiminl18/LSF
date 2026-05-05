def rule_page1_allcaps_ticker_headers(doc: dict) -> list[dict]:
    """Match all-caps short section headers on page 1 that are likely ticker symbols."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                if re.fullmatch(r"[A-Z]{2,8}(?:\d+[A-Z]{0,3})?", txt):
                    out.append(span)
        return out
    except Exception:
        return []
