def rule_nearby_after_anchor_headers(doc: dict) -> list[dict]:
    """Match spans within a short window after key financial headers."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        anchor_idx = []
        for i, s in enumerate(texts):
            if s.get("label") == "section_header" and re.search(r"(item\s*8|selected (financial|consolidated financial) data|statement of income|statement of operations|results of operations)", s.get("text", "") or "", re.I):
                anchor_idx.append(i)
        for i in anchor_idx:
            for j in range(i, min(i + 8, len(texts))):
                out.append(texts[j])
        return out
    except Exception:
        return []
