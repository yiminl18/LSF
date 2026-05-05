def rule_item1_business_listed_under_symbol(doc: dict) -> list[dict]:
    """Retrieve Item 1 Business spans that state stock is listed under a symbol."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            low = f"{path} {txt}".lower()
            if span.get("label") in {"text", "section_header"} and (
                "item 1" in low and "business" in low and "under the symbol" in low
            ):
                out.append(span)
        return out
    except Exception:
        return []

