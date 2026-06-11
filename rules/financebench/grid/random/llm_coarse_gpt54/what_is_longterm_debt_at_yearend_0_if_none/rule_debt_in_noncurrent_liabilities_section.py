def rule_debt_in_noncurrent_liabilities_section(doc: dict) -> list[dict]:
    """Match tables where debt appears under a non-current liabilities section."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            by_row = {}
            for c in cells:
                by_row.setdefault(c.get("row"), []).append((c.get("col"), c.get("text", "") or ""))
            rows = [(r, " | ".join(v for _, v in sorted(vals))) for r, vals in sorted(by_row.items())]
            seen_noncurrent = False
            for _, row_text in rows:
                if re.search(r"non[\-\s]?current liabilities|long[\-\s]?term liabilities", row_text, re.I):
                    seen_noncurrent = True
                elif seen_noncurrent and re.search(r"\bdebt\b|\blong[\-\s]?term debt\b", row_text, re.I):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
