def rule_same_path_as_target_heading(doc: dict) -> list[dict]:
    """Return spans sharing the same path_text as a heading/span that names the target table."""
    out = []
    try:
        target_paths = set()
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "maturity distribution" in txt and "average length" in txt:
                path = ((span.get("structure") or {}).get("path_text") or "")
                if path:
                    target_paths.add(path)
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "")
            if path in target_paths:
                out.append(span)
    except Exception:
        return []
    return out
