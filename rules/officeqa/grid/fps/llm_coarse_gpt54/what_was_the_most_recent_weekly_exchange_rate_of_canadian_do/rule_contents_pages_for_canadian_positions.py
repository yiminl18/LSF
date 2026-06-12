def rule_contents_pages_for_canadian_positions(doc: dict) -> list[dict]:
    """Match contents-page spans that point to Canadian dollar positions pages."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "contents" in path and "canadian dollar positions" in txt:
                out.append(span)
        return out
    except Exception:
        return []
