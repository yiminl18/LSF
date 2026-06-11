def rule_h1_path_text_company_not_form(doc: dict) -> list[dict]:
    """Match H1 section headers whose path_text looks like a company name rather than a form title."""
    try:
        out = []
        bad = ["FORM 10-", "CURRENT REPORT", "SECURITIES AND EXCHANGE COMMISSION", "WASHINGTON, D.C."]
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").upper()
            if (
                span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and span.get("page_no") == 1
                and not any(b in path for b in bad)
            ):
                out.append(span)
        return out
    except Exception:
        return []
