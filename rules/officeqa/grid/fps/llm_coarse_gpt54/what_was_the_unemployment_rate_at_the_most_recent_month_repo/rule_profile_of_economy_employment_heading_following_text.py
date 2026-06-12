def rule_profile_of_economy_employment_heading_following_text(doc: dict) -> list[dict]:
    """Match text spans immediately after the Employment and unemployment heading."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") == "section_header" and re.search(r"Employment and unemployment", span.get("text", "") or "", re.I):
                for j in range(i + 1, min(i + 6, len(texts))):
                    nxt = texts[j]
                    if nxt.get("label") == "section_header":
                        break
                    if nxt.get("label") == "text":
                        out.append(nxt)
    except Exception:
        return []
    return out
