def rule_exhibit_4_table_row(doc: dict) -> list[dict]:
    """Match exhibit tables containing a 4.x row with certificate/indenture/trust deed description."""
    import re
    out = []
    pat_num = re.compile(r'\b4(\.\d+)?[A-Za-z]?\b')
    pat_desc = re.compile(r'certificate|indenture|trust deed|notes due', re.I)
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
