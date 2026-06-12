def rule_profile_of_economy_consumer_sentiment(doc: dict) -> list[dict]:
    """Match spans in the Profile of the Economy section mentioning consumer sentiment or the Michigan/Reuters index."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "profile of the economy" in path.lower():
                if re.search(r"(consumer sentiment|university of michigan|michigan/reuters|reuters consumer sentiment)", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
