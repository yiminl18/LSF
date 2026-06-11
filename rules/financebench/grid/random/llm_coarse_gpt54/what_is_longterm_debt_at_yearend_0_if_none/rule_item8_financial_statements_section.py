def rule_item8_financial_statements_section(doc: dict) -> list[dict]:
    """Match spans under Item 8 / Financial Statements and Supplementary Data that mention debt."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure") or {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if re.search(r"item 8|financial statements|supplementary data|condensed consolidated", path, re.I):
                if re.search(r"\blong[\-\s]?term debt\b|\bdebt\b|\bborrowings\b", text, re.I) or span.get("label") == "table":
                    out.append(span)
        return out
    except Exception:
        return []
