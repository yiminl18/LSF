def rule_exhibit_10_table_row(doc: dict) -> list[dict]:
    """Match exhibit tables containing a 10.x row with agreement/plan description."""
    import re
    out = []
    pat_num = re.compile(r'\b10(\.\d+)?[A-Za-z]?\b')
    pat_desc = re.compile(r'agreement|plan|indenture|amendment|indemnity', re.I)
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c.get("text", "") or "")
            for vals in rows.values():
                joined = " | ".join(vals)
                if pat_num.search(joined) and pat_desc.search(joined):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
