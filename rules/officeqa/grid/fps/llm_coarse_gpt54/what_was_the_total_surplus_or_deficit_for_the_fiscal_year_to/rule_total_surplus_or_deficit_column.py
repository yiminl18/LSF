def rule_total_surplus_or_deficit_column(doc: dict) -> list[dict]:
    """Match tables containing a 'Total surplus or deficit' column."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "")
                cells = (((span.get("table_data") or {}).get("cells")) or [])
                if re.search(r'total surplus or deficit', txt, re.I):
                    out.append(span)
                    continue
                for c in cells:
                    if re.search(r'total surplus or deficit', c.get("text", ""), re.I):
                        out.append(span)
                        break
    except Exception:
        return []
    return out
