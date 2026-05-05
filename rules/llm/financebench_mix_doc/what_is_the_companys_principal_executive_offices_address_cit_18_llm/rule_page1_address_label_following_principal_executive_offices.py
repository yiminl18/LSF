def rule_page1_address_label_following_principal_executive_offices(doc: dict) -> list[dict]:
    """Match spans on page 1 immediately preceding or near '(Address of principal executive offices)' labels."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "address of principal executive offices" in txt:
                for j in range(max(0, i - 3), i):
                    cand = texts[j]
                    if cand.get("page_no") == span.get("page_no") == 1:
                        out.append(cand)
        return out
    except Exception:
        return []
