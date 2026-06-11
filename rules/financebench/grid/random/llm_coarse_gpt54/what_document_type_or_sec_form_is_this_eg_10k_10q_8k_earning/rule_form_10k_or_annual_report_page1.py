def rule_form_10k_or_annual_report_page1(doc: dict) -> list[dict]:
    """Match page-1 spans indicating Form 10-K either by form code or ANNUAL REPORT language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").upper()
            if re.search(r"\bFORM\s+10-K\b", t, re.I) or "ANNUAL REPORT" in t:
                out.append(span)
        return out
    except Exception:
        return []
