def rule_row_header_long_term_debt_cells(doc: dict) -> list[dict]:
    """Return table spans where any row header cell explicitly contains long-term debt."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            for c in span.get("table_data", {}).get("cells", []):
                if c.get("is_row_header") and re.search(r"\blong[- ]term debt\b", c.get("text") or "", re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
