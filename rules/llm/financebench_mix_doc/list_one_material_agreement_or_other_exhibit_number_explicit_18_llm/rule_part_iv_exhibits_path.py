def rule_part_iv_exhibits_path(doc: dict) -> list[dict]:
    """Match spans under Part IV paths that mention exhibits."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "").lower()
            if "part iv" in path and "exhibit" in path:
                out.append(span)
        return out
    except Exception:
        return []
