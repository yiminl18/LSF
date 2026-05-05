def rule_item1_overview_phone_fallback(doc: dict) -> list[dict]:
    """Match Item 1/Overview spans that mention the company's telephone number as a fallback pattern."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "").lower()
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if ("item 1" in path or "item 1" in text) and re.search(r"(telephone number|our telephone number)", text):
                out.append(span)
        return out
    except Exception:
        return []
