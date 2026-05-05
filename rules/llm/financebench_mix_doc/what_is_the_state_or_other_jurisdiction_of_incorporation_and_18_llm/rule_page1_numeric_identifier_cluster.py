def rule_page1_numeric_identifier_cluster(doc: dict) -> list[dict]:
    """Match page-1 spans with commission-file or EIN-like numeric identifiers near the cover header."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and (
                re.search(r"\b\d{2}-\d{7}\b", text)
                or re.search(r"\b\d+-\d+\b", text)
                or re.search(r"commission file", text, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
