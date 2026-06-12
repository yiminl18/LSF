def rule_neighbor_after_canadian_header(doc: dict) -> list[dict]:
    """Match the next few spans after a CANADIAN DOLLAR POSITIONS header, where the answer table usually sits."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "canadian dollar positions" in txt:
                for j in range(i + 1, min(i + 5, len(texts))):
                    out.append(texts[j])
        return out
    except Exception:
        return []
