def rule_esf_table_cells(doc: dict) -> list[dict]:
    """Match table spans whose structured cells mention ESF, balance sheet, or total assets."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for c in cells:
                t = c.get("text", "") or ""
                if re.search(r'\bESF-?1\b|Exchange Stabilization Fund|Balance sheet|Total assets', t, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
