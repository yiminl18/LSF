def rule_page_one_or_two_exhibit_mentions(doc: dict) -> list[dict]:
    """Match exhibit mentions on page 1-2, common in short 8-Ks."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if (span.get("page_no") or 0) in {1, 2}:
                text = span.get("text", "") or ""
                if re.search(r'Exhibit\s+\d+(\.\d+)?|\b99(\.\d+)?\b|\b104\b', text, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
