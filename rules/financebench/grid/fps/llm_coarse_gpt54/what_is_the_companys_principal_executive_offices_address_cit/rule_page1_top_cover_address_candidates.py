def rule_page1_top_cover_address_candidates(doc: dict) -> list[dict]:
    """Return broad page-1 cover-page address candidates near the registrant block."""
    import re
    try:
        out = []
        for idx, span in enumerate(doc.get("texts", [])):
            if span.get("page_no") != 1:
                continue
            if idx > 40:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'address of principal executive offices', txt, re.I):
                out.append(span)
                continue
            if re.search(r'\d{1,6}\s+\S+', txt) and (
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b', txt) or
                re.search(r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+', txt) or
                re.search(r'\bUnited Kingdom\b', txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
