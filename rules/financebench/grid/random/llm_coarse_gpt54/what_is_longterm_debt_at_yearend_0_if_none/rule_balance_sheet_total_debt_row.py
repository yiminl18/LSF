def rule_balance_sheet_total_debt_row(doc: dict) -> list[dict]:
    """Match balance sheet tables containing a 'Total debt' row, useful when long-term debt is summarized that way."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            if re.search(r"\btotal debt\b", text, re.I):
                out.append(span)
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), []).append(c.get("text", "") or "")
            for vals in row_texts.values():
                if re.search(r"\btotal debt\b", " | ".join(vals), re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
