def rule_item1_business_overview_located_at_address(doc: dict) -> list[dict]:
    """Match Item 1 Business spans with 'located at' and a full street address."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = (span.get("structure", {}).get("path_text") or "").lower()
            txt = span.get("text") or ""
            low = txt.lower()
            if "item 1" in path and "business" in path and "located at" in low:
                if re.search(r"\d{1,5}\s+[A-Za-z].+,\s*[A-Z][a-zA-Z ]+\s+\d{4,10}", txt):
                    out.append(span)
        return out
    except Exception:
        return []
