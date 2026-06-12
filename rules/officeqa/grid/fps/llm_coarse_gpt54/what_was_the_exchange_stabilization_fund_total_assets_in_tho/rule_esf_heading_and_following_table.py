def rule_esf_heading_and_following_table(doc: dict) -> list[dict]:
    """Match the first table span following an Exchange Stabilization Fund section header."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("label") == "section_header" and re.search(r'EXCHANGE STABILIZATION FUND', span.get("text", ""), re.I):
                for j in range(i + 1, min(i + 8, len(texts))):
                    nxt = texts[j]
                    if nxt.get("label") == "table":
                        out.append(nxt)
                        break
        return out
    except Exception:
        return []
