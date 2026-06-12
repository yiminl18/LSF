def rule_foreign_currency_path_with_fcp_codes(doc: dict) -> list[dict]:
    """Match spans under foreign currency positions whose text contains FCP codes."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "foreign currency positions" in path and "fcp-" in txt:
                out.append(span)
        return out
    except Exception:
        return []
