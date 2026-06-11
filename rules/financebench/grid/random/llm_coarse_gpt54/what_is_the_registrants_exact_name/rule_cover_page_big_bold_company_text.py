def rule_cover_page_big_bold_company_text(doc: dict) -> list[dict]:
    """Match big bold page-1 spans that are likely the registrant name, regardless of label."""
    try:
        out = []
        bad = ["FORM 10-", "CURRENT REPORT", "SECURITIES AND EXCHANGE COMMISSION", "WASHINGTON, D.C."]
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            if (
                span.get("page_no") == 1
                and span.get("bold") == 1
                and float(span.get("size") or 0) >= 12
                and len(txt) >= 3
                and not any(b in up for b in bad)
            ):
                out.append(span)
        return out
    except Exception:
        return []
