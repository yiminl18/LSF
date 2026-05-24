def rule_h1_company_cover_span_with_outstanding(doc: dict) -> list[dict]:
    """Match large company-name cover spans that embed the outstanding-share sentence."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            struct = span.get("structure", {}) or {}
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                if struct.get("level") == "H1" and "outstanding" in text.lower():
                    out.append(span)
        return out
    except Exception:
        return []
