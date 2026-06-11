def rule_page1_exchange_same_span_as_label(doc: dict) -> list[dict]:
    """Match page-1 spans where the label and exchange value are merged into one OCR span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        ex_re = re.compile(r"\b(new york stock exchange|the new york stock exchange|nasdaq|nasdaq global select market|the nasdaq global select market)\b")
        for span in texts:
            if span.get("page_no") == 1:
                low = (span.get("text") or "").lower()
                if "name of each exchange on which registered" in low and ex_re.search(low):
                    out.append(span)
        return out
    except Exception:
        return []
