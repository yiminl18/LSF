def rule_page1_h1_all_caps_company_name(doc: dict) -> list[dict]:
    """Match page-1 H1 company-name-like headers in all caps, excluding form/SEC headers."""
    try:
        out = []
        bad = ["FORM 10-", "SECURITIES AND EXCHANGE COMMISSION", "CURRENT REPORT", "PART I", "WASHINGTON, D.C."]
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and span.get("bold") == 1
                and len(txt) >= 3
                and not any(b in up for b in bad)
            ):
                out.append(span)
        return out
    except Exception:
        return []
