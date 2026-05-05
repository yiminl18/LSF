def rule_page1_company_header_with_embedded_phone(doc: dict) -> list[dict]:
    """Match company H1 headers whose text_span embeds the phone number in the cover block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1":
                text = span.get("text", "") or ""
                if re.search(r"form 10-|current report|commission", text, re.I):
                    continue
                ts = span.get("text_span", "") or ""
                if re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", ts):
                    out.append(span)
        return out
    except Exception:
        return []
