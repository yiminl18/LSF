def rule_tables_with_statement_of_financial_position(doc: dict) -> list[dict]:
    """Match tables for non-U.S. filers using 'statement of financial position' instead of balance sheet."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                text = (span.get("text") or "").lower()
                path = ((span.get("structure") or {}).get("path_text") or "").lower()
                if "statement of financial position" in text or "statement of financial position" in path:
                    out.append(span)
        return out
    except Exception:
        return []
