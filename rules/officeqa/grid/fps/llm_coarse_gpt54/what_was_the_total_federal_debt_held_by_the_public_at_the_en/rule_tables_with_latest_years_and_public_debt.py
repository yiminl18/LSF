def rule_tables_with_latest_years_and_public_debt(doc: dict) -> list[dict]:
    """Match tables that include recent year labels and public debt wording, useful in modern issues."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            txt = (span.get("text") or "").lower()
            if (
                ("2015" in txt or "2016" in txt or "2017" in txt or "2018" in txt or "2024" in txt)
                and ("debt held by the public" in txt or "held by the public" in txt or "summary of federal debt" in txt)
            ):
                out.append(span)
        return out
    except Exception:
        return []
