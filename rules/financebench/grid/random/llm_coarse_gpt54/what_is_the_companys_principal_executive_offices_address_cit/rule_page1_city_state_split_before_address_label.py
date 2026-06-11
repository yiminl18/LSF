def rule_page1_city_state_split_before_address_label(doc: dict) -> list[dict]:
    """Match page-1 spans containing city/state fragments that are followed nearby by the address label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'^[A-Z][A-Za-z .&\'-]+,\s*$', txt) or re.search(r'^[A-Z]{2,3}$', txt):
                window = " ".join((texts[j].get("text") or "") for j in range(i, min(i + 4, len(texts))))
                if re.search(r'address of principal executive offices', window, re.I):
                    out.append(span)
                    if i + 1 < len(texts):
                        out.append(texts[i + 1])
        return out
    except Exception:
        return []
