def rule_page1_form_or_report(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning FORM 10-K/10-Q/8-K or CURRENT REPORT."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I) or "CURRENT REPORT" in text.upper():
                out.append(span)
        return out
    except Exception:
        return []
