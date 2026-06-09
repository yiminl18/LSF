def rule_financial_statement_balance_sheet_debt_tables(doc: dict) -> list[dict]:
    """Match financial-statement balance-sheet tables with long-term debt rows, excluding Item 16 summary tables."""
    try:
        hits: list[dict] = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            text = " ".join((span.get("text") or "").split()).lower()
            path = " ".join((((span.get("structure") or {}).get("path_text")) or "").split()).lower()
            if "item 16" in path or "item 7a" in path or "market risk" in path:
                continue
            is_balance_sheet = (
                "balance sheet" in path
                or "balance sheets" in path
                or "financial position" in path
                or "condensed consolidated financial statements" in path
                or (
                    ("financial statements" in path or "financial information" in path)
                    and "assets" in text
                    and "liabilities" in text
                )
            )
            if not is_balance_sheet:
                continue
            if (
                "long-term debt" in text
                or "debt, non-current" in text
                or "long-term obligations" in text
                or ("long-term liabilities" in text and "debt" in text)
                or ("non-current liabilities" in text and "long-term debt" in text)
            ):
                hits.append(span)
        return hits
    except Exception:
        return []
