def rule_profile_of_economy_near_consumer_section(doc: dict) -> list[dict]:
    """Match spans near consumer-related section headers inside Profile of the Economy."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        idxs = set()
        for i, span in enumerate(texts):
            if span.get("label") == "section_header":
                text = (span.get("text") or "")
                path = ((span.get("structure") or {}).get("path_text") or "")
                if "profile of the economy" in path.lower() and re.search(r"(consumer|confidence|sentiment)", text, re.I):
                    for j in range(i, min(len(texts), i + 6)):
                        idxs.add(j)
        for j in sorted(idxs):
            out.append(texts[j])
        return out
    except Exception:
        return []
