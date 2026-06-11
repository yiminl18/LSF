def rule_debt_value_in_balance_sheet_current_long_term_pair(doc: dict) -> list[dict]:
    """Match balance sheet tables where both current debt and long-term debt values are shown in adjacent liability rows."""
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
            rows = [" | ".join(v for _, v in sorted(vals)) for _, vals in sorted(by_row.items())]
            has_current = any(re.search(r"current portion of long[\-\s]?term debt|short[\-\s]?term debt", r, re.I) for r in rows)
            has_long = any(re.search(r"long[\-\s]?term debt|long[\-\s]?term borrowings|long[\-\s]?term liabilities.*\bdebt\b", r, re.I) for r in rows)
            if has_current and has_long:
                out.append(span)
        return out
    except Exception:
        return []
