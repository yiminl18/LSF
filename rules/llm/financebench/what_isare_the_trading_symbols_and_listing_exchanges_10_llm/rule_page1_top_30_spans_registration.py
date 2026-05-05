def rule_page1_top_30_spans_registration(doc: dict) -> list[dict]:
    """Match registration-related spans among the first 30 spans, reflecting consistent top-of-document placement."""
    try:
        import re
        out = []
        for span in (doc.get("texts", [])[:30]):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"12\(b\)|trading symbol|exchange|registered|nasdaq|new york stock exchange|nyse", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
