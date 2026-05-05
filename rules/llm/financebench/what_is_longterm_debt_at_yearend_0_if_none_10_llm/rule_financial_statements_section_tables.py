def rule_financial_statements_section_tables(doc: dict) -> list[dict]:
    """Match tables under financial statements sections that mention debt or borrowings."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if ("financial statements" in path or "supplementary data" in path) and (
                re.search(r"\bdebt\b", txt) or re.search(r"\bborrowings\b", txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
