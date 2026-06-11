def rule_exhibit_rows_in_table_cells(doc: dict) -> list[dict]:
    """Return table spans whose cells contain exhibit-number-plus-description patterns."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            found_num = False
            found_desc = False
            for c in cells:
                t = c.get("text") or ""
                if re.search(r"\b(?:3|4|10|99|104)(?:\.\d+)?\b", t):
                    found_num = True
                if re.search(r"(agreement|plan|award|indenture|bylaws|press release|interactive data file)", t, re.I):
                    found_desc = True
            if found_num and found_desc:
                out.append(span)
    except Exception:
        return []
    return out
