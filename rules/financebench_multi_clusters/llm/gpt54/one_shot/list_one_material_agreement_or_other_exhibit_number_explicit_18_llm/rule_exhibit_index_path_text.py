def rule_exhibit_index_path_text(doc: dict) -> list[dict]:
    """Match spans whose structural path contains 'Exhibit Index'."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            if "exhibit index" in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
