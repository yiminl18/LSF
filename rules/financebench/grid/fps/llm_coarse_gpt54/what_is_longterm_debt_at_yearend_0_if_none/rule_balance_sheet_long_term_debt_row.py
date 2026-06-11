def rule_balance_sheet_long_term_debt_row(doc: dict) -> list[dict]:
    """Match balance sheet tables containing a row labeled long-term debt or long-term debt less current portion."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_text = {}
            for c in cells:
                row_text.setdefault(c.get("row"), {})[c.get("col")] = (c.get("text") or "").strip()
            for r, cols in row_text.items():
                label = " ".join(v for k, v in sorted(cols.items()) if k == 0).lower()
                if re.search(r"\blong[- ]term debt\b", label) or re.search(r"\blong[- ]term borrowings\b", label):
                    out.append(span)
                    break
                if "debt, less current portion" in label or "long-term debt, less current portion" in label:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
