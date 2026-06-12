def rule_fiscal_year_to_date_row(doc: dict) -> list[dict]:
    """Match tables containing a row labeled fiscal year to date / fiscal YYYY to date."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "")
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            if re.search(r'fiscal\s+(year|\d{4})\s+to\s+date', txt, re.I):
                out.append(span)
                continue
            for c in cells:
                if re.search(r'fiscal\s+(year|\d{4})\s+to\s+date', c.get("text", ""), re.I):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
