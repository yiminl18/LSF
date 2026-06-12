def rule_profile_of_economy_growth_sentence_with_same_as_previous_quarter(doc: dict) -> list[dict]:
    """Match GDP growth sentences that compare the latest quarter with prior quarters."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if re.search(r"real\s+GDP", txt, re.I) and re.search(r"(first|second|third|fourth)\s+quarter", txt, re.I) and re.search(r"(same as|compares with|following|after)", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
