def rule_annual_report_on_form_10k_for_fiscal_year(doc: dict) -> list[dict]:
    """Match spans containing the long annual-report phrase used by some 10-Ks."""
    try:
        return [
            span for span in doc.get("texts", [])
            if "annual report on form 10-k for the fiscal year ended" in ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
        ]
    except Exception:
        return []
