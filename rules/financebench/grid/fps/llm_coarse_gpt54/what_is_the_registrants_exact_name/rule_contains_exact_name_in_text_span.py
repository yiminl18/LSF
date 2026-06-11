def rule_contains_exact_name_in_text_span(doc: dict) -> list[dict]:
    """Match spans whose text_span contains the exact-name caption and use the span text as the answer."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            tspan = (span.get("text_span") or "").lower()
            if "exact name of registrant" in tspan:
                out.append(span)
        return out
    except Exception:
        return []
