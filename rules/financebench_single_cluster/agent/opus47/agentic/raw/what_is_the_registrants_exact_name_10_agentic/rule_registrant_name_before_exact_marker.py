def rule_registrant_name_before_exact_marker(doc: dict) -> list[dict]:
    """Span immediately preceding the '(Exact name of registrant ...)' charter-marker line on the 10-K cover page."""
    texts = doc.get("texts", [])
    results = []
    seen = set()
    for i, span in enumerate(texts):
        t = span.get("text", "")
        if "exact name of" in t.lower() and "as specified in its charter" in t.lower() and i > 0:
            prev = texts[i - 1]
            key = id(prev)
            if key in seen:
                continue
            seen.add(key)
            results.append(prev)
    return results
