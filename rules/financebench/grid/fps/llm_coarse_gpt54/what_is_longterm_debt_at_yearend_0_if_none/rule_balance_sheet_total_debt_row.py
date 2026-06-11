def rule_balance_sheet_total_debt_row(doc: dict) -> list[dict]:
    """Match tables with debt rows that often contain year-end long-term debt values, including total debt or principal amount of total debt."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            cells = span.get("table_data", {}).get("cells", [])
            row_labels = {}
            for c in cells:
                if c.get("col") == 0:
                    row_labels[c.get("row")] = (c.get("text") or "").strip().lower()
            for label in row_labels.values():
                if re.search(r"\btotal debt\b", label) or re.search(r"\bprincipal amount of total debt\b", label):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
