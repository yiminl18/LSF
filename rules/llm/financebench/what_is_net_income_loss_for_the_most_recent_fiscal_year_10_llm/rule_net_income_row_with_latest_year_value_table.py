def rule_net_income_row_with_latest_year_value_table(doc: dict) -> list[dict]:
    """Match tables where a net income row appears alongside multiple numeric year values."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c)
            for row_cells in rows.values():
                joined = " | ".join((c.get("text", "") or "") for c in sorted(row_cells, key=lambda x: x.get("col", 0)))
                if re.search(r"\bnet income\b|\bnet earnings\b|\bnet loss\b", joined, re.I):
                    nums = re.findall(r"\$?\(?\d[\d,]*\.?\d*\)?", joined)
                    if len(nums) >= 2:
                        out.append(span)
                        break
        return out
    except Exception:
        return []
