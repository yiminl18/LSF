def rule_tables_with_canadian_and_no_contents(doc: dict) -> list[dict]:
    """Match Canadian dollar tables excluding obvious contents pages."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if ("canadian dollar positions" in txt or "fcp-ii-" in txt) and "contents" not in path:
                out.append(span)
        return out
    except Exception:
        return []
