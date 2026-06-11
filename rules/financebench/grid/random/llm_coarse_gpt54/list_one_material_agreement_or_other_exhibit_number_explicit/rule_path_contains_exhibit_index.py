def rule_path_contains_exhibit_index(doc: dict) -> list[dict]:
    """Match any span whose structural path contains 'Exhibit Index'."""
    out = []
    try:
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if "exhibit index" in path.lower():
                out.append(span)
    except Exception:
        return []
    return out
