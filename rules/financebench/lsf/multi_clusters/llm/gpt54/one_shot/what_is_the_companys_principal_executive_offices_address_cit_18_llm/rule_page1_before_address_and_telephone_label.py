def rule_page1_before_address_and_telephone_label(doc: dict) -> list[dict]:
    """Match the immediately preceding span before '(Address and telephone number... principal executive offices)' labels."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "principal executive offices" in txt and "address and telephone number" in txt:
                if i - 1 >= 0 and texts[i - 1].get("page_no") == 1:
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
