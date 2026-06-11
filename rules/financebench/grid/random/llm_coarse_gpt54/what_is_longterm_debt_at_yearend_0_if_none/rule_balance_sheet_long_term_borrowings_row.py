def rule_balance_sheet_long_term_borrowings_row(doc: dict) -> list[dict]:
    """Match balance sheet tables where long-term debt is labeled as long-term borrowings."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"\blong[\-\s]?term borrowings\b", text, re.I):
                out.append(span)
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), []).append(c.get("text", "") or "")
            for vals in row_texts.values():
                if re.search(r"\blong[\-\s]?term borrowings\b", " | ".join(vals), re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
