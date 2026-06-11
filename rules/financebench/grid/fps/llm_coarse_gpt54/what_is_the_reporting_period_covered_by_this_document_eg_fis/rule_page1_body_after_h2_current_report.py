def rule_page1_body_after_h2_current_report(doc: dict) -> list[dict]:
    """Match body spans under H2 CURRENT REPORT paths that contain the event date."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("page_no") == 1 and "current report" in path:
                text = (span.get("text") or "").lower()
                if "date of report" in text or "event reported" in text:
                    out.append(span)
        return out
    except Exception:
        return []
