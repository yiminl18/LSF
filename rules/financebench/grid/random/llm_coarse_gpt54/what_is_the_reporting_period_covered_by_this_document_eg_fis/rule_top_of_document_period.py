def rule_top_of_document_period(doc: dict) -> list[dict]:
    """Match early spans in the document that contain the reporting period, regardless of label."""
    import re
    try:
        out = []
        for i, span in enumerate(doc.get("texts", [])[:25]):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'(fiscal year ended|quarterly period ended|Date of Report|earliest event reported)', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
