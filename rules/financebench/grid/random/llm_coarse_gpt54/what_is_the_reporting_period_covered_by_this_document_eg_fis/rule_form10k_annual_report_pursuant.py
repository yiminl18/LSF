def rule_form10k_annual_report_pursuant(doc: dict) -> list[dict]:
    """Match annual-report header spans that often carry the fiscal year ended date."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'ANNUAL REPORT PURSUANT TO SECTION 13 OR 15\(d\)', txt, re.I):
                if re.search(r'fiscal year ended', txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
