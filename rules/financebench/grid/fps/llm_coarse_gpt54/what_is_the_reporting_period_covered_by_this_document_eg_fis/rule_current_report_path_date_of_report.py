def rule_current_report_path_date_of_report(doc: dict) -> list[dict]:
    """Match spans under a CURRENT REPORT path that contain the event date line."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if "current report" in path and "date of report" in text:
                out.append(span)
        return out
    except Exception:
        return []
