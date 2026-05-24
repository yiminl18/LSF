def rule_item7_item8_financial_sections(doc: dict) -> list[dict]:
    """Retrieve broad Item 7/Item 8 financial sections and nearby statement-of-income content."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            level = ((span.get("structure") or {}).get("level") or "")
            hay = f"{path} {text}".lower()
            if (
                "item 7" in hay
                or "item 8" in hay
                or "management's discussion and analysis" in hay
                or "management’s discussion and analysis" in hay
                or "results of operations" in hay
                or "financial statements and supplementary data" in hay
                or "financial statements" in hay
                or "supplementary data" in hay
                or "statement of income" in hay
                or "statement of earnings" in hay
                or "statement of operations" in hay
                or "income before" in hay
                or "net earnings" in hay
                or "net income" in hay
                or "net loss" in hay
            ):
                out.append(span)
        return out
    except Exception:
        return []

