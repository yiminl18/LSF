def rule_page1_before_address_header(doc: dict) -> list[dict]:
    """Match the nearest preceding page-1 span before an address-of-principal-executive-offices caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "address of principal executive offices" in txt:
                for j in range(i - 1, max(-1, i - 5), -1):
                    if j >= 0 and texts[j].get("page_no") == 1:
                        cand = texts[j]
                        if any(ch.isalpha() for ch in (cand.get("text") or "")):
                            out.append(cand)
                            break
        return out
    except Exception:
        return []
