def rule_modern_fd2_path_or_text(doc: dict) -> list[dict]:
    """Match modern FD-2 debt-held-by-public tables by path or text."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "fd-2" in path and "debt held by the public" in path:
                out.append(span)
            elif "fd-2" in txt and "debt held by the public" in txt:
                out.append(span)
        return out
    except Exception:
        return []
