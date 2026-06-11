def rule_page1_allcaps_exchange_values(doc: dict) -> list[dict]:
    """Match short page-1 all-caps text spans that are likely exchange values such as NASDAQ."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("label") == "text":
                if txt in {"NASDAQ", "NYSE"} or txt == "NASDAQ GLOBAL SELECT MARKET":
                    out.append(span)
        return out
    except Exception:
        return []
