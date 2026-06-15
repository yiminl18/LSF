def rule_debt_note_longterm_table(doc: dict) -> list[dict]:
    """Match debt-note tables that summarize long-term debt totals or carrying values."""
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
            if "debt" not in path or any(key in path for key in ("cash", "investment", "securities")):
                continue
            if not any(
                key in text
                for key in (
                    "total long-term debt",
                    "long-term debt, excluding current portion",
                    "long-term debt, less current portion",
                    "long-term debt, net",
                    "carrying value of long-term debt",
                    "total non-current portion of term debt",
                    "total term debt",
                    "debt and obligations under finance leases",
                )
            ):
                continue
            if has_selected_total_term and any(key in path for key in ("term debt", "long-term debt")):
                continue
            if has_capital_structure_lease_debt and "obligations under finance leases" in text:
                continue
            out.append(s)
        return out
    except Exception:
        return []
