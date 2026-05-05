def rule_page1_cover_numeric_candidates_excluding_dollars(doc: dict) -> list[dict]:
    """Return large numeric page-1/2 spans excluding dollar-prefixed values."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no") in (1, 2) and re.fullmatch(r"[\d,]{6,}", text) and not text.startswith("$"):
                out.append(span)
        return out
    except Exception:
        return []
