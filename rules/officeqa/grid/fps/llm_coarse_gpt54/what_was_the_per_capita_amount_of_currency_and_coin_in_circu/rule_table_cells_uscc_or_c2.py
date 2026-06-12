def rule_table_cells_uscc_or_c2(doc: dict) -> list[dict]:
    """Match tables whose cells mention USCC/C-2 identifiers or per-capita comparative totals."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = (((span.get("table_data") or {}).get("cells")) or [])
            for c in cells:
                t = c.get("text") or ""
                if re.search(r'\b(USCC-?1|USCC-?2|C-?1|C-?2)\b', t, re.I) or re.search(r'per\s+capita\s+comparative\s+totals', t, re.I):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
