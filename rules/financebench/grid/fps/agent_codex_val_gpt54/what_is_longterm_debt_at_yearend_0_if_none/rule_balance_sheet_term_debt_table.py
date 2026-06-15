def rule_balance_sheet_term_debt_table(doc: dict) -> list[dict]:
    """Match balance-sheet tables that report Apple-style term debt rows."""
    try:
        import re

        term_row = re.compile(r"(^|\n)\|\s*(?:non-current portion of )?term debt\s*\|", re.I)
        return [
            s for s in doc.get("texts", [])
            if s.get("label") == "table"
            and any(
                key in (((s.get("structure") or {}).get("path_text") or "").lower())
                for key in ("balance sheet", "balance sheets")
            )
            and term_row.search(s.get("text") or "")
        ]
    except Exception:
        return []
