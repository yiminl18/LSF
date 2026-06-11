def rule_balance_sheet_notes_payable_and_long_term_debt(doc: dict) -> list[dict]:
    """Match balance sheet tables containing 'Notes payable and long-term debt' style labels."""
    try:
        import re
        out = []
        pat = r"notes payable.*long[\-\s]?term debt|long[\-\s]?term debt.*notes payable"
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(pat, text, re.I):
                out.append(span)
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), []).append(c.get("text", "") or "")
            for vals in row_texts.values():
                if re.search(pat, " | ".join(vals), re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
