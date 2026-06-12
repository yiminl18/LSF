def rule_old_ffo1_continuation_table(doc: dict) -> list[dict]:
    """Match old two-part FFO-1 continuation tables with selected balances and held-by-public formula column."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                "means of financing" in txt
                and "selected balances end of period" in txt
                and "held by the public" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
