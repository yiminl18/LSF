def rule_text_span_contains_address_label(doc: dict) -> list[dict]:
    """Match spans whose text_span, rather than text, contains the address label."""
    try:
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            tsp = (span.get("text_span") or "").lower()
            if "address of principal executive offices" in tsp:
                out.append(span)
            elif "address and telephone number, including area code, of registrant" in tsp:
                out.append(span)
            elif "address of principal executive offices and zip code" in tsp:
                out.append(span)
        return out
    except Exception:
        return []
