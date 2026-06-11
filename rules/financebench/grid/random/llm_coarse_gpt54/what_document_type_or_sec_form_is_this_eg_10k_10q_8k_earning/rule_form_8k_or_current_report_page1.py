def rule_form_8k_or_current_report_page1(doc: dict) -> list[dict]:
    """Match page-1 spans indicating Form 8-K either by form code or CURRENT REPORT language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            t = (span.get("text") or "").upper()
            if re.search(r"\bFORM\s+8-K\b", t, re.I) or "CURRENT REPORT" in t:
                out.append(span)
        return out
    except Exception:
        return []
