def rule_exhibit_table_cells(doc: dict) -> list[dict]:
    """Match table spans having cells with exhibit numbers or exhibit descriptions."""
    import re
    try:
        out = []
        num_pat = re.compile(r"^(?:exhibit\s*)?\d+(?:\.\d+)?[a-z]?$", re.I)
        desc_pat = re.compile(
            r"(credit agreement|transaction agreement|stock incentive plan|equity incentive plan|employee stock purchase plan|notice of stock option award|settlement agreement|bylaws|supplemental indenture|indenture|award)",
            re.I,
        )
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if any(num_pat.search((c.get("text") or "").strip()) for c in cells) and any(
                desc_pat.search((c.get("text") or "").strip()) for c in cells
            ):
                out.append(span)
        return out
    except Exception:
        return []
