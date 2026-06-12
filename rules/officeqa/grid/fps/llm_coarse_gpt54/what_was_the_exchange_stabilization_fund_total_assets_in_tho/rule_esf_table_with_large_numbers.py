def rule_esf_table_with_large_numbers(doc: dict) -> list[dict]:
    """Match ESF-related tables containing long digit strings likely to include the total-assets value."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text") or ""
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'ESF|Exchange Stabilization Fund', path + " " + text, re.I) and re.search(r'\b\d{6,}\b', text):
                out.append(span)
        return out
    except Exception:
        return []
