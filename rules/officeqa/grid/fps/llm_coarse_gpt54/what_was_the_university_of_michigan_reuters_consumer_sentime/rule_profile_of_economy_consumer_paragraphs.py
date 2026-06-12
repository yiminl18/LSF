def rule_profile_of_economy_consumer_paragraphs(doc: dict) -> list[dict]:
    """Match body text paragraphs in Profile of the Economy that discuss consumers and may include sentiment readings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "text":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "profile of the economy" in path.lower() and re.search(r"\bconsumer\b", text, re.I):
                out.append(span)
        return out
    except Exception:
        return []
