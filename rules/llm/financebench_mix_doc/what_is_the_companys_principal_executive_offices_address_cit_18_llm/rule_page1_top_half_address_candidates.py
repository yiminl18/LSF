def rule_page1_top_half_address_candidates(doc: dict) -> list[dict]:
    """Match early page 1 spans with address-like content in the top filing cover area."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for idx, span in enumerate(texts[:80]):
            if span.get("page_no") != 1:
                continue
            txt = span.get("text") or ""
            if re.search(r"\b[A-Z][a-zA-Z\.\- ]+,\s*(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)*)", txt):
                out.append(span)
            elif re.search(r"\b\d{1,5}\s+[A-Za-z].*(?:Plaza|Avenue|Boulevard|Drive|Road|Street|Center)", txt):
                out.append(span)
        return out
    except Exception:
        return []
