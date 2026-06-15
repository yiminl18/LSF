def rule_form8k_cover_item_headers(doc: dict) -> list[dict]:
    """Match compact 8-K cover and item-header spans for zero-debt event filings."""
    try:
        out = []
        for s in doc.get("texts", []):
            text = (s.get("text") or "").strip().lower()
            path = (((s.get("structure") or {}).get("path_text") or "")).lower()
            if s.get("page_no", 999) <= 2 and (
                text == "form 8-k"
                or text == "current report"
                or ("form 8-k" in path and text.startswith("item "))
            ):
                out.append(s)
        return out
    except Exception:
        return []
