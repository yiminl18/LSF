def rule_address_in_span_preceding_zip_label(doc: dict) -> list[dict]:
    """Match the span immediately before a ZIP code label span on page 1."""
    try:
        spans = doc.get("texts", [])
        out = []
        for i in range(1, len(spans)):
            cur = spans[i]
            if cur.get("page_no") != 1:
                continue
            txt = ((cur.get("text") or "") + " " + (cur.get("text_span") or "")).lower()
            if "(zip code)" in txt or "zip code" in txt:
                prev = spans[i - 1]
                if prev.get("page_no") == 1:
                    out.append(prev)
        return out
    except Exception:
        return []
