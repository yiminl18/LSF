def rule_profile_of_economy_unemployment_heading_following_text(doc: dict) -> list[dict]:
    """Match text spans immediately following an unemployment-related heading in the Profile of the Economy section."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            if span.get("label") != "section_header":
                continue
            header = span.get("text", "") or ""
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if not re.search(r"Employment and unemployment|Labor Markets|Unemployment Rate", header, re.I):
                continue
            if "Profile of the Economy" not in path and "Employment and unemployment" not in path and "Labor Markets" not in path and "Unemployment Rate" not in path:
                pass
            for j in range(i + 1, min(i + 6, len(texts))):
                nxt = texts[j]
                if nxt.get("label") == "section_header":
                    break
                if nxt.get("label") == "text":
                    out.append(nxt)
    except Exception:
        return []
    return out
