def rule_tables_with_debt_held_by_public_or_held_by_public(doc: dict) -> list[dict]:
    """Match any table containing either 'debt held by the public' or 'held by the public'."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "debt held by the public" in txt or "held by the public" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
