def rule_page1_h2_or_h3_short_metadata_headers(doc: dict) -> list[dict]:
    """Match short H2/H3 metadata headers on page 1 that often hold address fragments."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1 or span.get("label") != "section_header":
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            if level not in {"H2", "H3"}:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if re.search(r"\b\d{5}(?:-\d{4})?\b", txt):
                out.append(span)
            elif re.search(r"\b\d{1,6}\b", txt) and re.search(r"\b(avenue|drive|road|plaza|street|way)\b", low):
                out.append(span)
            elif txt in {"20817", "95125", "08933"}:
                out.append(span)
        return out
    except Exception:
        return []
