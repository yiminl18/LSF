def rule_item1_business_cover_sentence_with_city_state(doc: dict) -> list[dict]:
    """Match Item 1/Business text that directly states the city/state of principal offices."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").strip()
            if re.search(r'Item 1|Business', path, re.I) and (
                re.search(r'principal corporate offices are located in [A-Z][A-Za-z .-]+,\s*[A-Z][A-Za-z .-]+', txt) or
                re.search(r'executive offices .* located at .*?,\s*[A-Z][A-Za-z .-]+,\s*[A-Z][A-Za-z .-]+', txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
