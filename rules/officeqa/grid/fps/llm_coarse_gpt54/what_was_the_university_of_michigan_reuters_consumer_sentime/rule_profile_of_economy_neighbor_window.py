def rule_profile_of_economy_neighbor_window(doc: dict) -> list[dict]:
    """Match a broad neighbor window around any Profile of the Economy span mentioning consumer-related terms."""
    import re
    try:
        texts = doc.get("texts", [])
        idxs = set()
        for i, span in enumerate(texts):
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "profile of the economy" in path.lower() and re.search(r"(consumer|michigan|reuters|sentiment|confidence)", text, re.I):
                for j in range(max(0, i - 4), min(len(texts), i + 5)):
                    idxs.add(j)
        return [texts[j] for j in sorted(idxs)]
    except Exception:
        return []
