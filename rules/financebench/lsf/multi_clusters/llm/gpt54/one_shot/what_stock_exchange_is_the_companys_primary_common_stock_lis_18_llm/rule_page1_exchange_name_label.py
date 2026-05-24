def rule_page1_exchange_name_label(doc: dict) -> list[dict]:
    """Match page-1 spans containing the exchange-registration label text."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "name of each exchange" in txt and "registered" in txt:
                out.append(span)
        return out
    except Exception:
        return []
