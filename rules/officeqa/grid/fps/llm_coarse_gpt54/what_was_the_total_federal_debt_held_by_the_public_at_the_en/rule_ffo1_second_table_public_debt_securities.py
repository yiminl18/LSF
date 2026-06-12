def rule_ffo1_second_table_public_debt_securities(doc: dict) -> list[dict]:
    """Match the continuation FFO-1 table with selected balances including public debt securities and held by the public."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                "selected balances end of period" in txt
                and "public debt securities" in txt
                and "investments of government accounts" in txt
                and "held by the public" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
