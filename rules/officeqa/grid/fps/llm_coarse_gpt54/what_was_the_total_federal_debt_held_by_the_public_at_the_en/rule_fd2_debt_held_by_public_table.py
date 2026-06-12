def rule_fd2_debt_held_by_public_table(doc: dict) -> list[dict]:
    """Match FD-2 tables named Debt Held by the Public in modern Treasury Bulletins."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "fd-2" in txt and "debt held by the public" in txt:
                out.append(span)
            elif "debt held by the public" in path:
                out.append(span)
        return out
    except Exception:
        return []
