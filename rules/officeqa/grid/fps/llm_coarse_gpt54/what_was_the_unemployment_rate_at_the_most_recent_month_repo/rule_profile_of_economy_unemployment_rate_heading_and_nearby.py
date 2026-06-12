def rule_profile_of_economy_unemployment_rate_heading_and_nearby(doc: dict) -> list[dict]:
    """Match the Unemployment Rate heading and nearby text/figure spans."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = span.get("text", "") or ""
            if span.get("label") in {"section_header", "text", "caption"} and re.search(r"^Unemployment Rate$", text, re.I):
                out.append(span)
                for j in range(i + 1, min(i + 5, len(texts))):
                    nxt = texts[j]
                    if nxt.get("label") == "section_header":
                        break
                    out.append(nxt)
    except Exception:
        return []
    return out
