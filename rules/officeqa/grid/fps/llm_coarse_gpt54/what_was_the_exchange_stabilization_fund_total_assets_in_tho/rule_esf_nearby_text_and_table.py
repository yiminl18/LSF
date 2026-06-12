def rule_esf_nearby_text_and_table(doc: dict) -> list[dict]:
    """Match tables near text spans mentioning Exchange Stabilization Fund or ESF-1."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if re.search(r'Exchange Stabilization Fund|\bESF-?1\b', span.get("text", "") or "", re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 4)):
                    if texts[j].get("label") == "table":
                        out.append(texts[j])
        # dedupe preserving order
        seen = set()
        dedup = []
        for s in out:
            key = id(s)
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        return dedup
    except Exception:
        return []
