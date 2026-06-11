def rule_item_601_exhibits_table_rows(doc: dict) -> list[dict]:
    """Match exhibit-list tables by finding rows/cells with exhibit-style numbering and descriptions."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            texts = [c.get("text", "") for c in cells]
            joined = " ".join(texts)
            if re.search(r"\b(?:4|10|99|104)\.\d+\b", joined) or re.search(r"\b104\b", joined):
                if re.search(r"(agreement|plan|award|indenture|bylaws|press release|interactive data file)", joined, re.I):
                    out.append(span)
    except Exception:
        return []
    return out
