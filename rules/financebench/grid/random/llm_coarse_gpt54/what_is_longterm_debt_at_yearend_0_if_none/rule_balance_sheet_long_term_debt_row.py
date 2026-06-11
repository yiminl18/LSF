def rule_balance_sheet_long_term_debt_row(doc: dict) -> list[dict]:
    """Match balance sheet table spans containing a 'Long-term debt' row, often the direct answer location."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = span.get("text", "") or ""
            path = (span.get("structure") or {}).get("path_text", "") or ""
            if not (
                re.search(r"balance sheet", text, re.I)
                or re.search(r"balance sheet", path, re.I)
                or re.search(r"financial statements", path, re.I)
            ):
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_texts = {}
            for c in cells:
                row_texts.setdefault(c.get("row"), []).append((c.get("col"), c.get("text", "") or ""))
            for row, vals in row_texts.items():
                row_join = " | ".join(v for _, v in sorted(vals))
                if re.search(r"\blong[\-\s]?term debt\b", row_join, re.I):
                    out.append(span)
                    break
            if re.search(r"\blong[\-\s]?term debt\b", text, re.I):
                if span not in out:
                    out.append(span)
        return out
    except Exception:
        return []
