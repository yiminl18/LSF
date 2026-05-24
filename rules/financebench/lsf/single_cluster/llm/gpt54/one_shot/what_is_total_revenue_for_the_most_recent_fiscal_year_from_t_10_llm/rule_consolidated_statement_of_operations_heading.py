def rule_consolidated_statement_of_operations_heading(doc: dict) -> list[dict]:
    """Match section headers naming Consolidated Statement(s) of Operations."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "section_header":
                continue
            txt = (span.get("text") or "").lower()
            if "consolidated statement of operations" in txt or "consolidated statements of operations" in txt:
                out.append(span)
        return out
    except Exception:
        return []
