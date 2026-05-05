def rule_report_type_in_first_page_first_40(doc: dict) -> list[dict]:
    """Match report-type spans in the first 40 spans on page 1."""
    import re
    try:
        out = []
        count = 0
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            count += 1
            if count > 40:
                break
            blob = ((span.get("text") or "") + " " + (span.get("text_span") or "")).upper()
            if (
                re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", blob, re.I)
                or "CURRENT REPORT" in blob
                or "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(D)" in blob
                or "QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(D)" in blob
            ):
                out.append(span)
        return out
    except Exception:
        return []
