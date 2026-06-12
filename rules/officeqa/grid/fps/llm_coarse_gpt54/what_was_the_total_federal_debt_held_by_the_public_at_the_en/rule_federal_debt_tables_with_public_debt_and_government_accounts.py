def rule_federal_debt_tables_with_public_debt_and_government_accounts(doc: dict) -> list[dict]:
    """Match tables containing public debt securities, government accounts, and held-by-public style balance logic."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                "public debt securities" in txt
                and "investments of government accounts" in txt
                and ("held by the public" in txt or "18+19-20" in txt or "18+19-20" in txt.replace(" ", ""))
            ):
                out.append(span)
        return out
    except Exception:
        return []
