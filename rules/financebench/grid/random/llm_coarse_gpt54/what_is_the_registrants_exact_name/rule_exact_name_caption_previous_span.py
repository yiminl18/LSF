def rule_exact_name_caption_previous_span(doc: dict) -> list[dict]:
    """Match the span immediately preceding a caption containing 'Exact name of registrant as specified in its charter'."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "exact name of registrant as specified in its charter" in txt:
                if i > 0:
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
