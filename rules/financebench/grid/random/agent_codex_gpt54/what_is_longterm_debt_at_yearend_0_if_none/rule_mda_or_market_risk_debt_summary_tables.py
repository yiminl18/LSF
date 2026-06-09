def rule_mda_or_market_risk_debt_summary_tables(doc: dict) -> list[dict]:
    """Match MD&A or market-risk tables summarizing debt, net debt, or long-term debt totals."""
    try:
        hits: list[dict] = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = " ".join((span.get("text") or "").split()).lower()
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            if not any(
                key in path
                for key in (
                    "management's discussion",
                    "management’s discussion",
                    "reconciliation of net debt",
                )
            ):
                continue
            if (
                "total net carrying amount" in text
                or "reconciliation of net debt" in path
                or "long-term debt, less current portion" in text
                or "total debt |" in text
                or "total debt " in text
                or "total gross long-term debt" in text
            ):
                hits.append(span)
        return hits
    except Exception:
        return []
