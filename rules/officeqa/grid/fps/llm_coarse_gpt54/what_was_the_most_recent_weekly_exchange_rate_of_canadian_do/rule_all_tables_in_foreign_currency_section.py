def rule_all_tables_in_foreign_currency_section(doc: dict) -> list[dict]:
    """Match all table spans whose path_text is within the foreign currency positions section."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                path = ((span.get("structure") or {}).get("path_text") or "").lower()
                if "foreign currency positions" in path:
                    out.append(span)
        return out
    except Exception:
        return []
