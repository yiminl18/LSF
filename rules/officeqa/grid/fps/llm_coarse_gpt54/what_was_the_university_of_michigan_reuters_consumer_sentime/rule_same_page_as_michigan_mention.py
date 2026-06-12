def rule_same_page_as_michigan_mention(doc: dict) -> list[dict]:
    """Match all spans on pages where Michigan/Reuters consumer sentiment is mentioned."""
    import re
    try:
        texts = doc.get("texts", [])
        pages = set()
        for span in texts:
            text = (span.get("text") or "")
            if re.search(r"(university of michigan|michigan/reuters|consumer sentiment|reuters consumer sentiment)", text, re.I):
                if span.get("page_no") is not None:
                    pages.add(span.get("page_no"))
        return [span for span in texts if span.get("page_no") in pages]
    except Exception:
        return []
