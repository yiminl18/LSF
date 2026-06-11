def rule_cover_page_report_keyword_only(doc: dict) -> list[dict]:
    """Match page-1 report-type spans even if the exact form code is split elsewhere."""
    try:
        keys = [
            "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)",
            "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)",
            "CURRENT REPORT",
        ]
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1:
                t = (span.get("text") or "").upper()
                if any(k in t for k in keys):
                    out.append(span)
        return out
    except Exception:
        return []
