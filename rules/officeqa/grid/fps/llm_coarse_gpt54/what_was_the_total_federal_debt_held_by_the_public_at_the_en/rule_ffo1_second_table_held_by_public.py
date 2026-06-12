def rule_ffo1_second_table_held_by_public(doc: dict) -> list[dict]:
    """Match FFO-1 table spans whose text contains the 'Held by the public' selected-balances column."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                path = ((span.get("structure") or {}).get("path_text") or "").lower()
                if "ffo-1" in path and "held by the public" in txt:
                    out.append(span)
                elif "summary of fiscal operations" in path and "held by the public" in txt:
                    out.append(span)
                elif "held by the public" in txt and "selected balances end of period" in txt:
                    out.append(span)
        return out
    except Exception:
        return []
