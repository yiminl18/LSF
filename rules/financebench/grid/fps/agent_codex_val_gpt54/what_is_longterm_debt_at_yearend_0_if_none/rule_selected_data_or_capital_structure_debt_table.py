def rule_selected_data_or_capital_structure_debt_table(doc: dict) -> list[dict]:
    """Match selected-data and capital-structure tables that summarize year-end debt."""
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("label") == "table"
            and any(
                key in (((s.get("structure") or {}).get("path_text") or "").lower())
                for key in (
                    "selected financial data",
                    "liquidity and capital resources",
                    "capital structure",
                    "capital resources",
                    "off-balance sheet arrangements",
                    "five-year summary of selected financial data",
                )
            )
            and any(
                key in (s.get("text") or "").lower()
                for key in (
                    "total term debt",
                    "long-term debt",
                    "long term debt",
                    "long-term debt and obligations under finance leases",
                )
            )
        ]
    except Exception:
        return []
