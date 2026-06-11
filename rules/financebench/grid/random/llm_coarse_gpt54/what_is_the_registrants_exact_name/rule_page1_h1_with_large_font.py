def rule_page1_h1_with_large_font(doc: dict) -> list[dict]:
    """Match page-1 H1 section headers with unusually large font, typical of the registrant name."""
    try:
        out = []
        for span in doc.get("texts", []):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and float(span.get("size") or 0) >= 13
                and "FORM 10-" not in (span.get("text") or "").upper()
                and "SECURITIES AND EXCHANGE COMMISSION" not in (span.get("text") or "").upper()
            ):
                out.append(span)
        return out
    except Exception:
        return []
