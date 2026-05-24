def rule_page1_shares_outstanding_text_only(doc: dict) -> list[dict]:
    """Match page-1 text-label spans that mention shares outstanding even if the number is separate."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            t = (span.get("text") or "").lower()
            if "shares" in t and "outstanding" in t:
                out.append(span)
    except Exception:
        return []
    return out
