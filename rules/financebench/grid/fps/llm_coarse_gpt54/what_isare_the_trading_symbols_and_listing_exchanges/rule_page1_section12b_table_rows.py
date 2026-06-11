def rule_page1_section12b_table_rows(doc: dict) -> list[dict]:
    """Match page-1 table spans with row data under Section 12(b) registration."""
    import re
    try:
        out = []
        for s in doc.get("texts", []):
            if s.get("label") != "table" or s.get("page_no") != 1:
                continue
            cells = (((s.get("table_data") or {}).get("cells")) or [])
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), []).append(c.get("text") or "")
            for _, vals in row_texts.items():
                joined = " | ".join(vals)
                if re.search(r"Common Stock|Notes Due|par value", joined, re.I) and re.search(r"NASDAQ|NYSE|New York Stock Exchange|Nasdaq", joined, re.I):
                    out.append(s)
                    break
        return out
    except Exception:
        return []
