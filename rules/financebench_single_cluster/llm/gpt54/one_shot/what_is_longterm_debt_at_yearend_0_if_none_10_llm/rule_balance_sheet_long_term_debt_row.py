def rule_balance_sheet_long_term_debt_row(doc: dict) -> list[dict]:
    """Match balance sheet tables containing a row labeled long-term debt or long-term debt excluding current maturities."""
    import re
    try:
        out = []
        texts = doc.get("texts", [])
        for span in texts:
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if not (
                "balance sheet" in txt
                or "balance sheet" in path
                or "financial statements" in path
            ):
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_text = {}
            for c in cells:
                row_text.setdefault(c.get("row"), []).append((c.get("col"), c.get("text", "")))
            for r, vals in row_text.items():
                vals_sorted = [t for _, t in sorted(vals)]
                row_join = " | ".join(vals_sorted).lower()
                if re.search(r"\blong[\-\s]?term debt\b", row_join) or re.search(r"\blong[\-\s]?term borrowings\b", row_join):
                    out.append(span)
                    break
                if "debt excluding current maturities" in row_join:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
