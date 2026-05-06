def rule_page1_address_and_telephone_principal_offices(doc: dict) -> list[dict]:
    """Match page 1 spans preceding '(Address and telephone number, including area code, of registrant’s principal executive offices)'."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "principal executive offices" in txt and "telephone number" in txt:
                for j in range(max(0, i - 3), i):
                    cand = texts[j]
                    if cand.get("page_no") == 1:
                        out.append(cand)
        return out
    except Exception:
        return []
