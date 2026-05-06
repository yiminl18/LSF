def rule_registrant_phone_cover_page(doc: dict) -> list[dict]:
    """Retrieve page-1 cover-page spans containing the registrant telephone number."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"text", "section_header", "list_item"}:
                continue
            txt = span.get("text", "") or ""
            path = ((span.get("structure") or {}).get("path_text", "") or "")
            low = txt.lower()
            path_low = path.lower()
            if (
                "telephone number" in low
                or "including area code" in low
                or "address and telephone number" in low
                or "telephone number" in path_low
                or re.search(r"\+\d{1,3}\s*\d{2,4}[\s\-]?\d{3,}", txt)
                or re.search(r"\(\d{3}\)\s*\d{3}[\-\)]\d{4}", txt)
                or re.search(r"\b\d{3}[\-]\d{3}[\-]\d{4}\b", txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []

