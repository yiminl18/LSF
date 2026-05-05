def rule_path_contains_item8(doc: dict) -> list[dict]:
    """Match any span under a path containing Item 8 and Financial Statements."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            if "item 8" in path and "financial statements" in path:
                out.append(span)
        return out
    except Exception:
        return []
