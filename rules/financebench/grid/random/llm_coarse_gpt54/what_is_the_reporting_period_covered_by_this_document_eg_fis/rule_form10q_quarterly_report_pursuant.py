def rule_form10q_quarterly_report_pursuant(doc: dict) -> list[dict]:
    """Match quarterly-report header spans that often carry the quarterly period ended date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15\(d\)', txt, re.I):
                if re.search(r'quarterly period ended', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
