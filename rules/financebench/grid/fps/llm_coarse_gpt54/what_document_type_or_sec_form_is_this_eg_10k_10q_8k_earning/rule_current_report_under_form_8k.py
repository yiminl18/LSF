def rule_current_report_under_form_8k(doc: dict) -> list[dict]:
    """Match CURRENT REPORT spans whose path is under FORM 8-K."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip().upper()
            path = (span.get("structure", {}).get("path_text") or "").upper()
            if "CURRENT REPORT" in text and "FORM 8-K" in path:
                out.append(span)
        return out
    except Exception:
        return []
