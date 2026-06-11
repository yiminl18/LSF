def rule_long_term_debt_numeric_cell_row(doc: dict) -> list[dict]:
    """Return table spans where a row header mentions long-term debt and numeric value cells appear in the same row."""
    try:
        import re
        out = []
        num_pat = re.compile(r"^\$?\s*\(?\d[\d,\.]*\)?\s*$")
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            by_row = {}
            for c in cells:
                by_row.setdefault(c.get("row"), []).append(c)
            for row, vals in by_row.items():
                row_text = " | ".join((c.get("text", "") or "") for c in vals)
                if re.search(r"\blong[\-\s]?term debt\b", row_text, re.I):
                    if any(num_pat.match((c.get("text", "") or "").strip()) for c in vals):
                        out.append(span)
                        break
        return out
    except Exception:
        return []
