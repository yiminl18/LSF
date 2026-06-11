def rule_exhibit_table_rows_material_types(doc: dict) -> list[dict]:
    """Match exhibit tables containing rows with common material-agreement descriptions."""
    import re
    out = []
    kws = [
        r'credit agreement', r'indenture', r'supplemental indenture',
        r'employment agreement', r'wafer supply agreement',
        r'executive incentive plan', r'deferred compensation plan',
        r'indemnity agreement', r'employee stock purchase plan',
        r'press release', r"officer'?s certificate", r'trust deed'
    ]
    pat = re.compile("|".join(kws), re.I)
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            row_text = {}
            for c in cells:
                row_text.setdefault(c.get("row"), []).append(c.get("text", "") or "")
            for r, vals in row_text.items():
                joined = " | ".join(vals)
                if pat.search(joined):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
