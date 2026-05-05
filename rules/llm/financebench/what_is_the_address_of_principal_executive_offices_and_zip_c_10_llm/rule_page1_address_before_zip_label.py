def rule_page1_address_before_zip_label(doc: dict) -> list[dict]:
    """Match spans immediately before a ZIP-code label span."""
    try:
        spans = doc.get("texts", [])
        out = []
        for i, span in enumerate(spans):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "(zip code)" in txt or "zip code" in txt:
                for j in range(max(0, i - 2), i):
                    if spans[j].get("page_no") == 1:
                        out.append(spans[j])
        return out
    except Exception:
        return []
