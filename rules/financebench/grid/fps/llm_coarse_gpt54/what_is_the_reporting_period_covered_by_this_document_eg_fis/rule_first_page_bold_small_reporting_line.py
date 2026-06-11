def rule_first_page_bold_small_reporting_line(doc: dict) -> list[dict]:
    """Match page-1 bold spans in small/medium font that contain reporting-period language."""
    try:
        out = []
        for span in doc.get("texts", []):
            size = span.get("size") or 0
            if span.get("page_no") == 1 and span.get("bold") == 1 and size <= 12.5:
                text = (span.get("text") or "").lower()
                if any(k in text for k in [
                    "fiscal year ended",
                    "quarterly period ended",
                    "date of report",
                    "date of earliest event reported"
                ]):
                    out.append(span)
        return out
    except Exception:
        return []
