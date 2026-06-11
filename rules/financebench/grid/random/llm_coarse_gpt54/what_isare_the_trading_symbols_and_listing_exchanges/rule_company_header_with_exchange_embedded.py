def rule_company_header_with_exchange_embedded(doc: dict) -> list[dict]:
    """Match page-1 company header spans whose text or text_span contains exchange-registration language."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            blob = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).strip()
            if span.get("page_no") == 1 and span.get("label") == "section_header" and (
                re.search(r"name of each exchange on which registered", blob, re.I)
                or re.search(r"new york stock exchange|nasdaq", blob, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
