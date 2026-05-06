def rule_page1_before_exchange_header(doc: dict) -> list[dict]:
    """Match spans immediately before a 'Name of each exchange...' header on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if span.get("page_no") == 1 and "name of each exchange" in txt:
                for j in range(max(0, i - 4), i + 1):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
