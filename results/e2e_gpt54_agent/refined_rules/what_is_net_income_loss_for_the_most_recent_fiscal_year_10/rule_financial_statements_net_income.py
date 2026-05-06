def rule_financial_statements_net_income(doc: dict) -> list[dict]:
    """Retrieve financial statement or MD&A spans likely containing net income for the latest fiscal year."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            label = span.get("label") or ""
            hay = f"{path} {text}".lower()
            in_fin_section = (
                "financial statements" in hay
                or "financial statement" in hay
                or "supplementary data" in hay
                or "management's discussion and analysis" in hay
                or "management’s discussion and analysis" in hay
                or "results of operations" in hay
                or "statement of income" in hay
                or "statement of earnings" in hay
                or "statement of operations" in hay
                or "income statement" in hay
                or "consolidated results of operations" in hay
            )
            mentions_metric = (
                "net income" in hay
                or "net earnings" in hay
                or "net loss" in hay
                or "net income (loss)" in hay
                or "net earnings (loss)" in hay
            )
            if in_fin_section and (mentions_metric or label == "table"):
                out.append(span)
        return out
    except Exception:
        return []

