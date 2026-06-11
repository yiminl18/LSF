def rule_page1_h2_or_h3_address_header(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 section headers that are address lines in the registrant cover block."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            if level not in {"H2", "H3", "H4", "H5"}:
                continue
            text = (span.get("text") or "").strip()
            if re.search(r'\d{1,6}\s+\S+', text) and (
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', text) or
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+', text) or
                re.search(r'\bUnited Kingdom\b', text)
            ):
                out.append(span)
        return out
    except Exception:
        return []
