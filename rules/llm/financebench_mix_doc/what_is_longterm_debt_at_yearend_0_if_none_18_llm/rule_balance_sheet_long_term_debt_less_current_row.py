def rule_balance_sheet_long_term_debt_less_current_row(doc: dict) -> list[dict]:
    """Match balance sheet table spans containing 'long-term debt, less current portion'."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_text = {}
            for c in cells:
                row_text.setdefault(c.get("row"), []).append((c.get("col"), c.get("text", "")))
            for r, vals in row_text.items():
                joined = " ".join(v for _, v in sorted(vals)).lower()
                if re.search(r"long[- ]term debt.*less current portion", joined):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
