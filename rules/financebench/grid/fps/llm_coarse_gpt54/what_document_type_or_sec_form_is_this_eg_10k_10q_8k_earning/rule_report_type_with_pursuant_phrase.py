def rule_report_type_with_pursuant_phrase(doc: dict) -> list[dict]:
    """Match report-type spans that include the standard 'Pursuant to Section 13 or 15(d)' phrase."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combo = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"Pursuant to Section 13 or 15\(d\)", combo, re.I):
                out.append(span)
        return out
    except Exception:
        return []
