def rule_exact_name_sibling(doc: dict) -> list[dict]:
    """Return the span immediately before 'Exact name of Registrant' label on page 1."""
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = span.get("text", "").lower()
            if "exact name of registrant" in text and i > 0:
                prev = texts[i - 1]
                if prev.get("page_no") == 1 and prev.get("bold") == 1:
                    return [prev]
        return []
    except Exception:
        return []
