def rule_page1_h2_or_h3_phone_header(doc: dict) -> list[dict]:
    """Match H2/H3 page-1 headers whose text or text_span contains a phone number and telephone label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            lvl = span.get("structure", {}).get("level")
            if span.get("page_no") == 1 and lvl in {"H2", "H3"}:
                blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
                if re.search(r"(telephone number|area code)", blob, re.I) and re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                    out.append(span)
        return out
    except Exception:
        return []
