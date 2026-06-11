def rule_exhibit_description_keywords_anywhere(doc: dict) -> list[dict]:
    """Match spans containing common exhibit-description keywords even without the word Exhibit."""
    import re
    out = []
    keywords = r"(credit agreement|transaction agreement|settlement agreement|stock incentive plan|equity incentive plan|employee stock purchase plan|notice of stock option award|second supplemental indenture|bylaws as amended|press release dated|cover page interactive data file)"
    try:
        for span in doc.get("texts", []):
            if re.search(keywords, span.get("text") or "", re.I):
                out.append(span)
    except Exception:
        return []
    return out
