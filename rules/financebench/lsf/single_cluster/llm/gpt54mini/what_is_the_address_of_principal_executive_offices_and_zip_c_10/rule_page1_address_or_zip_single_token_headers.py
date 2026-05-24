def rule_page1_address_or_zip_single_token_headers(doc: dict) -> list[dict]:
    """Match single-token page-1 headers that are ZIP codes or short address fragments."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r"\d{5}(?:-\d{4})?", txt):
                out.append(span)
            elif len(txt.split()) <= 6 and re.search(r"\b\d{1,6}\b", txt):
                out.append(span)
        return out
    except Exception:
        return []
