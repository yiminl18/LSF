def rule_zip_code_label(doc: dict) -> list[dict]:
    """Return the ZIP code span preceding the (Zip Code) label on page 1."""
    try:
        texts = doc.get("texts", [])
        for i, s in enumerate(texts):
            if s.get("page_no", 0) > 2:
                continue
            text_lower = s.get("text", "").lower()
            if "(zip code)" == text_lower.strip():
                if i > 0:
                    return [texts[i-1], s]
        return []
    except Exception:
        return []
