def rule_page1_first_20_spans_period(doc: dict) -> list[dict]:
    """Match reporting-period phrases appearing very early in the document."""
    import re
    try:
        out = []
        for i, span in enumerate(doc.get("texts", [])[:20]):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if re.search(r'(for the fiscal year ended|for the quarterly period ended|date of report \(date of earliest event reported\))', txt):
                out.append(span)
        return out
    except Exception:
        return []
