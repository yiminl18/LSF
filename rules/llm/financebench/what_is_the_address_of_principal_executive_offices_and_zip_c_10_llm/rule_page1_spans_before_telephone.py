def rule_page1_spans_before_telephone(doc: dict) -> list[dict]:
    """Return page-1 spans immediately preceding the registrant telephone number."""
    try:
        spans = doc.get("texts", [])
        out = []
        for i, span in enumerate(spans):
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "telephone number" in txt:
                for j in range(max(0, i - 3), i):
                    if spans[j].get("page_no") == 1:
                        out.append(spans[j])
        return out
    except Exception:
        return []
