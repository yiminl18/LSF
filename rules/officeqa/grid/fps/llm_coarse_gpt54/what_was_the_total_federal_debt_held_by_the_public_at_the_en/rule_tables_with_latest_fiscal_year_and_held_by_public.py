def rule_tables_with_latest_fiscal_year_and_held_by_public(doc: dict) -> list[dict]:
    """Match tables containing both fiscal-year rows and a held-by-public column."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") == "table":
                txt = (span.get("text") or "").lower()
                if "held by the public" in txt and (
                    "fiscal year" in txt or "fiscal 198" in txt or "fiscal 199" in txt or "fiscal 20" in txt
                ):
                    out.append(span)
        return out
    except Exception:
        return []
