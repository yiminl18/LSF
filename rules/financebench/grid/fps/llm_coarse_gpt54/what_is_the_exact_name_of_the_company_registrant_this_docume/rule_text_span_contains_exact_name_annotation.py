def rule_text_span_contains_exact_name_annotation(doc: dict) -> list[dict]:
    """Match spans whose text_span contains the exact-name-of-registrant annotation and return the span itself."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            ts = (span.get("text_span", "") or "").lower()
            if "exact name of registrant as specified in its charter" in ts:
                out.append(span)
        return out
    except Exception:
        return []
