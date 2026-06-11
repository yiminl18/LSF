def rule_exchange_in_cover_span_with_multiple_metadata_fields(doc: dict) -> list[dict]:
    """Match dense cover spans that bundle company metadata and exchange registration text."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            combined = " ".join([span.get("text", "") or "", span.get("text_span", "") or ""])
            if span.get("page_no") == 1 and re.search(r'commission file|employer identification|address of principal executive offices', combined, re.I):
                if re.search(r'new york stock exchange|nasdaq|global select market', combined, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
