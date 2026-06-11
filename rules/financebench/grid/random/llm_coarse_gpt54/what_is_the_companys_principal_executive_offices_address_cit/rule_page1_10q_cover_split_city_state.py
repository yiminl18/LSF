def rule_page1_10q_cover_split_city_state(doc: dict) -> list[dict]:
    """Match split city/state components on page 1 in 10-Q covers."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'^[A-Z][A-Za-z .-]+,\s*$', txt) or re.fullmatch(r'CA|WA|NY', txt):
                window = " ".join((texts[j].get("text") or "") for j in range(max(0, i - 2), min(len(texts), i + 3)))
                if re.search(r'Address of principal executive offices', window, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
