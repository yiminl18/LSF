def rule_form_code_or_report_type_in_header_page(doc: dict) -> list[dict]:
    """Match spans on their header_page containing either a form code or report-type phrase."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            t = (span.get("text") or "").upper()
            if span.get("page_no") == span.get("header_page", span.get("page_no")):
                if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", t, re.I) or any(k in t for k in ["ANNUAL REPORT", "QUARTERLY REPORT", "CURRENT REPORT"]):
                    out.append(span)
        return out
    except Exception:
        return []
