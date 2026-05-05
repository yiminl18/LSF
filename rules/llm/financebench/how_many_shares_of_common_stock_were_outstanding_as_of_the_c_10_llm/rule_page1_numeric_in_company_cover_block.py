def rule_page1_numeric_in_company_cover_block(doc: dict) -> list[dict]:
    """Match large numeric spans on page 1 under company-name cover paths, useful for split-label layouts."""
    import re
    out = []
    try:
        num_re = re.compile(r"^\d[\d,]{5,}$")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").strip()
            if num_re.match(txt) and path and "item 1" not in path:
                out.append(span)
    except Exception:
        return []
    return out
