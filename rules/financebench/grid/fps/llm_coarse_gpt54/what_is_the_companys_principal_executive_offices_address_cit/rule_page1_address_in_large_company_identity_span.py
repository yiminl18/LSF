def rule_page1_address_in_large_company_identity_span(doc: dict) -> list[dict]:
    """Match large-font page-1 company identity spans that embed the address in text_span."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if (span.get("size") or 0) < 10:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r'\d{1,6}\s+\S+.*\b[A-Z][a-z]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+)', txt):
                out.append(span)
        return out
    except Exception:
        return []
