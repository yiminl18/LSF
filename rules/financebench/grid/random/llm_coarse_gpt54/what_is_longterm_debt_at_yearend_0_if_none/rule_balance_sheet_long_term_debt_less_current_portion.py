def rule_balance_sheet_long_term_debt_less_current_portion(doc: dict) -> list[dict]:
    """Match balance sheet tables with the specific row 'Long-term debt, less current portion'."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"long[\-\s]?term debt,\s*less current portion", text, re.I):
                out.append(span)
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), []).append(c.get("text", "") or "")
            for vals in row_texts.values():
                row_join = " | ".join(vals)
                if re.search(r"long[\-\s]?term debt,\s*less current portion", row_join, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
