def rule_statement_of_financial_position_phrase(doc: dict) -> list[dict]:
    """Match spans whose text/path contains statement of financial position."""
    try:
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "statement of financial position" in text or "statement of financial position" in path:
                out.append(span)
        return out
    except Exception:
        return []
