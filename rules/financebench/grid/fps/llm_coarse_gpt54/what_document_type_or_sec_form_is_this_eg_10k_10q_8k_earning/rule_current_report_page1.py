def rule_current_report_page1(doc: dict) -> list[dict]:
    """Match page-1 spans with CURRENT REPORT, useful for 8-K filings."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip().upper()
            if span.get("page_no") == 1 and "CURRENT REPORT" in text:
                out.append(span)
        return out
    except Exception:
        return []
