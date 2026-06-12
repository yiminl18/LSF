def rule_section_and_following_table(doc: dict) -> list[dict]:
    """Return currency/coin section headers and the next nearby table span after them."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if span.get("label") == "section_header" and re.search(r'currency\s+and\s+coin', txt, re.I):
                out.append(span)
                for j in range(i + 1, min(i + 8, len(texts))):
                    nxt = texts[j]
                    if nxt.get("label") == "table":
                        out.append(nxt)
                        break
    except Exception:
        return []
    return out
