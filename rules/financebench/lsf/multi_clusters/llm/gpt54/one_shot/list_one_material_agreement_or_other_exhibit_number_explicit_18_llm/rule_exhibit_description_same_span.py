def rule_exhibit_description_same_span(doc: dict) -> list[dict]:
    """Match spans that contain both an exhibit number and a likely exhibit title in the same span."""
    import re
    try:
        out = []
        num_pat = re.compile(r"\b(?:exhibit\s+)?\d+(?:\.\d+)?[a-z]?\b", re.I)
        title_pat = re.compile(
            r"(credit agreement|transaction agreement|stock incentive plan|equity incentive plan|employee stock purchase plan|notice of stock option award|settlement agreement|bylaws|supplemental indenture|indenture|press release|cover page interactive data file)",
            re.I,
        )
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if num_pat.search(txt) and title_pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
