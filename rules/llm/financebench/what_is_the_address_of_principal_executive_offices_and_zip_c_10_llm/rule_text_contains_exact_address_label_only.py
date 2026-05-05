def rule_text_contains_exact_address_label_only(doc: dict) -> list[dict]:
    """Match spans whose text is exactly the address label, often adjacent to the answer."""
    try:
        spans = doc.get("texts", [])
        out = []
        targets = {
            "(address of principal executive offices)",
            "(address of principal executive offices and zip code)",
            "(address and telephone number, including area code, of registrant’s principal executive offices)",
            "(address and telephone number, including area code, of registrant's principal executive offices)",
        }
        for span in spans:
            txt = (span.get("text") or "").strip().lower()
            if txt in targets:
                out.append(span)
        return out
    except Exception:
        return []
