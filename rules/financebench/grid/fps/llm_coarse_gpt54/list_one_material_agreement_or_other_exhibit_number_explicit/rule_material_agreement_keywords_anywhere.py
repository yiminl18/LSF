def rule_material_agreement_keywords_anywhere(doc: dict) -> list[dict]:
    """Match spans mentioning common material-agreement exhibit types."""
    import re
    out = []
    kws = [
        r'credit agreement', r'indenture', r'supplemental indenture',
        r'employment agreement', r'wafer supply agreement',
        r'executive incentive plan', r'deferred compensation plan',
        r'indemnity agreement', r'stock purchase plan', r'employee stock purchase plan',
        r'merger agreement', r'plan of merger', r'notes due', r'trust deed'
    ]
    pat = re.compile("|".join(kws), re.I)
    try:
        for span in doc.get("texts", []):
            text = span.get("text", "") or ""
            if pat.search(text):
                out.append(span)
    except Exception:
        return []
    return out
