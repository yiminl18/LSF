def rule_address_in_text_span_followed_by_address_label(doc: dict) -> list[dict]:
    """Match a text span immediately followed by an address label span."""
    try:
        spans = doc.get("texts", [])
        out = []
        for i in range(len(spans) - 1):
            a, b = spans[i], spans[i + 1]
            if a.get("page_no") != 1 or b.get("page_no") != 1:
                continue
            ta = (a.get("text") or "").strip()
            tb = ((b.get("text") or "") + " " + (b.get("text_span") or "")).lower()
            if "address of principal executive offices" in tb or "address of principal executive offices and zip code" in tb:
                out.append(a)
        return out
    except Exception:
        return []
