def rule_item_15_exhibits_path(doc: dict) -> list[dict]:
    """Match spans under a path mentioning Item 15 and Exhibits."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "").lower()
            if "item 15" in path and "exhibit" in path:
                out.append(span)
        return out
    except Exception:
        return []
