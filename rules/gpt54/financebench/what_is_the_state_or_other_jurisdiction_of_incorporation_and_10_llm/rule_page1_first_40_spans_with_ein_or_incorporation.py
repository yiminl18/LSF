def rule_page1_first_40_spans_with_ein_or_incorporation(doc: dict) -> list[dict]:
    """Match early page-1 spans near the top of the filing containing EIN/incorporation cues."""
    try:
        import re
        out = []
        for i, span in enumerate(doc.get("texts", [])[:40]):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"\d{2}-\d{7}|incorporation|jurisdiction|employer identification", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
