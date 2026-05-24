def rule_form_heading_or_report_in_section_header_text_span(doc: dict) -> list[dict]:
    """Match section headers whose text_span contains report-type language."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "section_header":
                continue
            blob = (s.get("text_span") or "").upper()
            if (
                re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", blob, re.I)
                or "CURRENT REPORT" in blob
                or "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in blob
                or "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)" in blob
            ):
                out.append(s)
        return out
    except Exception:
        return []
