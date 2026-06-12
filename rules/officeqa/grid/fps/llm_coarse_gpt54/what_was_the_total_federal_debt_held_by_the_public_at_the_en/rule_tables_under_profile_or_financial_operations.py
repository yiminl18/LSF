def rule_tables_under_profile_or_financial_operations(doc: dict) -> list[dict]:
    """Match tables under Profile of the Economy or Financial Operations where summary debt values often appear in later vintages."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "profile of the economy" in path or "financial operations" in path:
                out.append(span)
        return out
    except Exception:
        return []
