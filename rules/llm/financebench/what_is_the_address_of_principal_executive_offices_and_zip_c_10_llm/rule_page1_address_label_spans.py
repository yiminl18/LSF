def rule_page1_address_label_spans(doc: dict) -> list[dict]:
    """Match page-1 spans containing the principal executive office address label."""
    try:
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            low = txt.lower()
            if "address of principal executive offices" in low:
                out.append(span)
            elif "address and telephone number, including area code, of registrant’s principal executive offices" in low:
                out.append(span)
            elif "address and telephone number, including area code, of registrant's principal executive offices" in low:
                out.append(span)
            elif "address of principal executive offices and zip code" in low:
                out.append(span)
        return out
    except Exception:
        return []
