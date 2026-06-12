def rule_2024_profile_page7_consumer_sentiment(doc: dict) -> list[dict]:
    """Match page 7 Profile of the Economy spans in 2024-style documents where the answer appears."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            page = span.get("page_no")
            path = ((span.get("structure") or {}).get("path_text") or "")
            text = (span.get("text") or "")
            if page == 7 and "profile of the economy" in path.lower():
                if re.search(r"(consumer|sentiment|confidence|michigan|reuters)", text, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
