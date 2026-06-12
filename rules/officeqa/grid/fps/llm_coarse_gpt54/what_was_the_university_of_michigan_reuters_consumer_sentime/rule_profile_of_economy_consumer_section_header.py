def rule_profile_of_economy_consumer_section_header(doc: dict) -> list[dict]:
    """Match section headers in Profile of the Economy that may introduce consumer sentiment/confidence discussion."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "profile of the economy" in path.lower() and re.search(r"(consumer|confidence|sentiment)", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
