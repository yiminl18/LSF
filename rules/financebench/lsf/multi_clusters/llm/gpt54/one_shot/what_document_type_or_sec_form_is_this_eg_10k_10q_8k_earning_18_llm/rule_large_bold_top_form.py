def rule_large_bold_top_form(doc: dict) -> list[dict]:
    """Match large bold top-of-page form/report headings on page 1."""
    import re
    try:
        out = []
        for span in doc.get("texts", [])[:20]:
            text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and span.get("bold") == 1 and float(span.get("size") or 0) >= 10:
                if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", text, re.I) or "CURRENT REPORT" in text.upper():
                    out.append(span)
        return out
    except Exception:
        return []
