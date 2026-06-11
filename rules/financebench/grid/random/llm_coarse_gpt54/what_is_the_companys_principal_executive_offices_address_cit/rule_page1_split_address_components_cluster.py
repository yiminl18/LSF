def rule_page1_split_address_components_cluster(doc: dict) -> list[dict]:
    """Match clusters of adjacent page-1 spans that together form split address components."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 2):
            a = texts[i]
            b = texts[i + 1]
            c = texts[i + 2]
            if a.get("page_no") == b.get("page_no") == c.get("page_no") == 1:
                combo = " ".join([(a.get("text") or "").strip(), (b.get("text") or "").strip(), (c.get("text") or "").strip()])
                if re.search(r'[A-Z][A-Za-z .-]+,\s*(?:[A-Z]{2}|California|Washington|Minnesota|New York)', combo) or re.search(r'Bristol.*United Kingdom', combo, re.I):
                    out.extend([a, b, c])
        return out
    except Exception:
        return []
