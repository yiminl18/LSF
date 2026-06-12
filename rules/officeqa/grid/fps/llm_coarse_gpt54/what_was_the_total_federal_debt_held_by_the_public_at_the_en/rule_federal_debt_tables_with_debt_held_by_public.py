def rule_federal_debt_tables_with_debt_held_by_public(doc: dict) -> list[dict]:
    """Match Federal Debt tables explicitly mentioning 'debt held by the public'."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                path = ((span.get("structure") or {}).get("path_text") or "").lower()
                if "debt held by the public" in txt and "federal debt" in path:
                    out.append(span)
                elif "debt held by the public" in txt and "fd-" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
