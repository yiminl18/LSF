def rule_form_and_report_type_same_span(doc: dict) -> list[dict]:
    """Match spans that mention both FORM 10-K/10-Q/8-K and ANNUAL/QUARTERLY/CURRENT REPORT language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").upper()
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I):
                if any(k in text for k in ["ANNUAL REPORT", "QUARTERLY REPORT", "CURRENT REPORT"]):
                    out.append(span)
        return out
    except Exception:
        return []
