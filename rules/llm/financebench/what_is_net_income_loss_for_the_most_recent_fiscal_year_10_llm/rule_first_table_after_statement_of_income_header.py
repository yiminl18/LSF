def rule_first_table_after_statement_of_income_header(doc: dict) -> list[dict]:
    """Match the first table after a Statement of Income/Operations header."""
    import re
    try:
        texts = doc.get("texts", [])
        for i, s in enumerate(texts):
            if s.get("label") == "section_header" and re.search(r"(statement|statements) of (income|operations|earnings)", s.get("text", "") or "", re.I):
                for j in range(i + 1, min(i + 10, len(texts))):
                    if texts[j].get("label") == "table":
                        return [texts[j]]
        return []
    except Exception:
        return []
