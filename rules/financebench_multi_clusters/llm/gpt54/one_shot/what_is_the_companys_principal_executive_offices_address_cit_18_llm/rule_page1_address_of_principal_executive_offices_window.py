def rule_page1_address_of_principal_executive_offices_window(doc: dict) -> list[dict]:
    """Match a wider window around the principal executive offices label on page 1 for high recall."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            tsp = (span.get("text_span") or "").lower()
            if "principal executive offices" in txt or "principal executive offices" in tsp:
                for j in range(max(0, i - 5), min(len(texts), i + 3)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
