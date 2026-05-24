def rule_long_term_debt_in_financial_text(doc: dict) -> list[dict]:
    """Match any financial-looking text span with long-term debt regardless of exact section."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") not in {"text", "section_header", "list_item"}:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if re.search(r"\blong[\-\s]?term debt\b", txt) and (
                "item 7" in path or "item 8" in path or "financial" in path or "debt" in path
            ):
                out.append(span)
        return out
    except Exception:
        return []
