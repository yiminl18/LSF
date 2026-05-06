def rule_page1_spans_before_phone(doc: dict) -> list[dict]:
    """Match page-1 spans immediately preceding telephone-number spans."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "telephone number" in txt or re.search(r"\(\d{3}\)\s*\d{3}[- ]?\d{4}", txt):
                for j in range(max(0, i - 2), i):
                    if spans[j].get("page_no") == 1:
                        out.append(spans[j])
        return out
    except Exception:
        return []
