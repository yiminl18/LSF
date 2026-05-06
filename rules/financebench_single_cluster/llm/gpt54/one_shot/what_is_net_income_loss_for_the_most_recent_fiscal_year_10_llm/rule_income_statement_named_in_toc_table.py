def rule_income_statement_named_in_toc_table(doc: dict) -> list[dict]:
    """Match TOC tables that explicitly list Statement of Income/Operations pages."""
    import re
    try:
        return [
            s for s in doc.get("texts", [])
            if s.get("label") == "table"
            and re.search(r"(consolidated )?(statement|statements) of (income|operations|earnings)", s.get("text", "") or "", re.I)
        ]
    except Exception:
        return []
