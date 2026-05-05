def rule_cover_page_report_phrase_after_form(doc: dict) -> list[dict]:
    """Match report-type phrases near the form heading on page 1."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").upper()
            if (
                "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in txt
                or "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)" in txt
                or "CURRENT REPORT" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
