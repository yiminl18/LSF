def rule_material_agreement_keywords(doc: dict) -> list[dict]:
    """Match spans containing common material-agreement exhibit titles like credit agreement, stock plan, indenture, settlement agreement, bylaws, award."""
    import re
    try:
        out = []
        kws = [
            r"credit agreement",
            r"transaction agreement",
            r"stock incentive plan",
            r"equity incentive plan",
            r"employee stock purchase plan",
            r"notice of stock option award",
            r"settlement agreement",
            r"bylaws",
            r"supplemental indenture",
            r"indenture",
            r"award",
        ]
        pat = re.compile("|".join(kws), re.I)
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if pat.search(txt):
                out.append(span)
        return out
    except Exception:
        return []
