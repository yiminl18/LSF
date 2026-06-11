def rule_current_report_form_8k_phrase(doc: dict) -> list[dict]:
    """Match spans containing both CURRENT REPORT and FORM 8-K phrasing."""
    try:
        out = []
        for span in doc.get("texts", []):
            t = (span.get("text") or "").upper()
            if "CURRENT REPORT" in t and "FORM 8-K" in t:
                out.append(span)
        return out
    except Exception:
        return []
