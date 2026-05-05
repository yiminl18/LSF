def rule_page1_spans_after_ein(doc: dict) -> list[dict]:
    """Match page-1 spans near the I.R.S. Employer Identification No. label."""
    try:
        spans = doc.get("texts", [])
        out = []
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "employer identification no" in txt or "i.r.s. employer identification no" in txt:
                for j in range(max(0, i - 3), min(len(spans), i + 4)):
                    if spans[j].get("page_no") == 1:
                        out.append(spans[j])
        return out
    except Exception:
        return []
