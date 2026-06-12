def rule_profile_economy_deficit_sentence_on_right_column(doc: dict) -> list[dict]:
    """Match Profile of the Economy deficit/GDP spans often appearing in right-column text on later issues."""
    out = []
    try:
        for span in doc.get("texts", []):
            page = span.get("page_no")
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if (
                isinstance(page, int) and page in {9, 10, 11}
                and span.get("label") == "text"
                and "profile of the economy" in path
                and "deficit" in txt
                and "gdp" in txt
            ):
                out.append(span)
    except Exception:
        return []
    return out
