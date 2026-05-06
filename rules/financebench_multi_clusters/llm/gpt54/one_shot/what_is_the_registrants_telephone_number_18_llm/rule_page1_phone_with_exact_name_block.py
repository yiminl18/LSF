def rule_page1_phone_with_exact_name_block(doc: dict) -> list[dict]:
    """Match page-1 spans in the exact-name block that also contain a phone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if span.get("page_no") == 1 and re.search(r"exact name of registrant", path + " " + blob, re.I) and re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
