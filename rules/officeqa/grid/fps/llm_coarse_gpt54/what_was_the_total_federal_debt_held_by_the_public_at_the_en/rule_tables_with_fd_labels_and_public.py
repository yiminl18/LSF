def rule_tables_with_fd_labels_and_public(doc: dict) -> list[dict]:
    """Match FD-labeled tables that also mention public debt/public holdings."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if ("fd-1" in txt or "fd-2" in txt or "fd-3" in txt) and "public" in txt:
                out.append(span)
        return out
    except Exception:
        return []
