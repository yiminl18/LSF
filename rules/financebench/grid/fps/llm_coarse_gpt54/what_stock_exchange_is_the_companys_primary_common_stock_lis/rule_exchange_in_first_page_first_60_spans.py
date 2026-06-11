def rule_exchange_in_first_page_first_60_spans(doc: dict) -> list[dict]:
    """Match exchange-name spans among the first 60 page-1 spans."""
    import re
    try:
        out = []
        count = 0
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                count += 1
                txt = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
                if re.search(r'new york stock exchange|the nasdaq global select market|the nasdaq stock market llc|NASDAQ\b', txt, re.I):
                    out.append(span)
                if count >= 60:
                    break
        return out
    except Exception:
        return []
