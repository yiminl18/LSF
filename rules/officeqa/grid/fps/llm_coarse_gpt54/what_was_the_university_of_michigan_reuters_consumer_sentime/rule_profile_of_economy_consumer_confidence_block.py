def rule_profile_of_economy_consumer_confidence_block(doc: dict) -> list[dict]:
    """Match spans in Profile of the Economy discussing consumer confidence/sentiment, including nearby numeric spans."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        hit_idxs = set()
        for i, span in enumerate(texts):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "profile of the economy" in path.lower() and re.search(r"(consumer confidence|consumer sentiment|michigan|reuters)", text, re.I):
                for j in range(max(0, i - 2), min(len(texts), i + 3)):
                    hit_idxs.add(j)
        for j in sorted(hit_idxs):
            out.append(texts[j])
        return out
    except Exception:
        return []
