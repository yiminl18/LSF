def rule_table_row_with_common_stock_and_exchange(doc: dict) -> list[dict]:
    """Match table spans containing a row with common stock and an exchange value."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append((c.get("col"), c.get("text") or ""))
            for _, vals in rows.items():
                row_text = " | ".join(v for _, v in sorted(vals))
                if re.search(r'common stock', row_text, re.I) and re.search(
                    r'new york stock exchange|nasdaq|global select market', row_text, re.I
                ):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
