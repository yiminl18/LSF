def rule_page1_form_section_reporting_lines(doc: dict) -> list[dict]:
    """Match page-1 spans under FORM 10-K/10-Q/8-K sections that contain reporting-period language."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "form " in path:
                if any(k in text for k in [
                    "fiscal year ended",
                    "quarterly period ended",
                    "quarter ended",
                    "date of report",
                    "date of earliest event reported",
                    "event reported",
                    "transition period"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
