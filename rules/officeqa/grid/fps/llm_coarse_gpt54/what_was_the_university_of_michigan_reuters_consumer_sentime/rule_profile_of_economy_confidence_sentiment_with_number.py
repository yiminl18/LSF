def rule_profile_of_economy_confidence_sentiment_with_number(doc: dict) -> list[dict]:
    """Match Profile of the Economy spans containing confidence/sentiment plus a likely index number."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if "profile of the economy" in path.lower():
                if re.search(r"(confidence|sentiment)", text, re.I) and re.search(r"\b\d{2,3}\.?\d*\b", text):
                    out.append(span)
        return out
    except Exception:
        return []
