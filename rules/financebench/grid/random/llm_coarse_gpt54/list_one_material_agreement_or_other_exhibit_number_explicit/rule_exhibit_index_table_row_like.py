def rule_exhibit_index_table_row_like(doc: dict) -> list[dict]:
    """Match tables likely to be exhibit indexes by dense legal-document exhibit descriptions."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = span.get("text") or ""
            score = 0
            if re.search(r"\bexhibit", txt, re.I):
                score += 1
            if re.search(r"\bdescription\b", txt, re.I):
                score += 1
            if re.search(r"\b(?:agreement|plan|indenture|bylaws|press release|interactive data file)\b", txt, re.I):
                score += 1
            if re.search(r"\b(?:3|4|10|99|104)(?:\.\d+)?\b", txt):
                score += 1
            if score >= 2:
                out.append(span)
    except Exception:
        return []
    return out
