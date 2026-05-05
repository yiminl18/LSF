def rule_balance_sheet_long_term_debt_and_current_portion(doc: dict) -> list[dict]:
    """Match balance sheet tables that contain both long-term debt and current portion/current maturities debt rows."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            rows = {}
            for c in cells:
                rows.setdefault(c.get("row"), []).append(c.get("text", ""))
            has_long = False
            has_current = False
            for vals in rows.values():
                row = " | ".join(vals).lower()
                if re.search(r"\blong[\-\s]?term debt\b", row) or re.search(r"\blong[\-\s]?term borrowings\b", row):
                    has_long = True
                if "current maturities" in row or "current portion" in row:
                    has_current = True
            if has_long and has_current:
                out.append(span)
        return out
    except Exception:
        return []
