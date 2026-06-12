def rule_profile_economy_deficit_sentence_on_left_column(doc: dict) -> list[dict]:
    """Match Profile of the Economy deficit/GDP spans often appearing in left-column text on some issues."""
    out = []
    try:
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                isinstance(page, int) and page in {7, 8, 9, 10, 11}
                and span.get("label") == "text"
                and "profile of the economy" in path
                and ("budget deficit" in txt or "federal deficit" in txt)
                and "gdp" in txt
            ):
                out.append(span)
    except Exception:
        return []
    return out
