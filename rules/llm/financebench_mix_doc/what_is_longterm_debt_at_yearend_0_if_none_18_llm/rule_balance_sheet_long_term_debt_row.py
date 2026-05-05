def rule_balance_sheet_long_term_debt_row(doc: dict) -> list[dict]:
    """Match balance sheet table spans containing a long-term debt row."""
    import re
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if not (
                "balance sheet" in text
                or "balance sheets" in text
                or "consolidated balance sheet" in text
                or "condensed consolidated balance sheet" in text
                or "financial statements" in path
            ):
                continue
            cells = ((span.get("table_data") or {}).get("cells") or [])
            row_text = {}
            for c in cells:
                row_text.setdefault(c.get("row"), []).append((c.get("col"), c.get("text", "")))
            for r, vals in row_text.items():
                joined = " ".join(v for _, v in sorted(vals)).lower()
                if re.search(r"\blong[- ]term debt\b", joined):
                    out.append(span)
                    break
    except Exception:
        return []
    return out
