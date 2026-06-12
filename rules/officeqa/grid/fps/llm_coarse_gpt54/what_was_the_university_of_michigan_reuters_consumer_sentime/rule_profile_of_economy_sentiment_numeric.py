def rule_profile_of_economy_sentiment_numeric(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans that mention sentiment and contain a likely numeric reading."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "profile of the economy" not in path.lower():
                continue
            if re.search(r"(sentiment|confidence)", text, re.I) and re.search(r"\b\d{2,3}\.?\d*\b", text):
                out.append(span)
        return out
    except Exception:
        return []
