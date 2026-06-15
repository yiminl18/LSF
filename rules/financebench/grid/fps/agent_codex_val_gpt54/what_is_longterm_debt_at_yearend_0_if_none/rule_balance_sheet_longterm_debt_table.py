def rule_balance_sheet_longterm_debt_table(doc: dict) -> list[dict]:
    """Match balance-sheet style tables with explicit long-term debt rows."""
    try:
        texts = doc.get("texts", [])
        has_selected_total_term = any(
            span.get("label") == "table"
            and "selected financial data" in (((span.get("structure") or {}).get("path_text") or "").lower())
            and "total term debt" in (span.get("text") or "").lower()
            for span in texts
        )
        has_capital_structure_lease_debt = any(
            span.get("label") == "table"
            and "capital structure" in (((span.get("structure") or {}).get("path_text") or "").lower())
            and any(
                key in (span.get("text") or "").lower()
                for key in (
                    "long-term debt and obligations under finance leases",
                    "long-term debt and obligations under capital leases",
                )
            )
            for span in texts
        )

        out = []
        for s in texts:
            if s.get("label") != "table":
                continue
            path = (((s.get("structure") or {}).get("path_text") or "")).lower()
            text = (s.get("text") or "").lower()
            if not any(key in path for key in ("balance sheet", "balance sheets", "financial position", "capitalization")):
                continue
            if not any(
                key in text
                for key in (
                    "long-term debt",
                    "long term debt",
                    "long-term borrowings",
                    "long term borrowings",
                    "current maturities of long-term debt",
                    "obligations under finance leases",
                )
            ):
                continue
            if has_selected_total_term and "current portion of long-term debt" in text and "long-term debt" in text:
                continue
            if has_capital_structure_lease_debt and "obligations under finance leases" in text:
                continue
            out.append(s)
        return out
    except Exception:
        return []
