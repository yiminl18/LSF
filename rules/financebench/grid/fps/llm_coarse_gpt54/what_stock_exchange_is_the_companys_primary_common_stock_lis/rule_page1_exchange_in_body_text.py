def rule_page1_exchange_in_body_text(doc: dict) -> list[dict]:
    """Match page-1 body/text spans whose text is or contains the exchange name."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("label") == "text":
                txt = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
                if re.search(r'new york stock exchange|the nasdaq global select market|the nasdaq stock market llc|NASDAQ\b', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
