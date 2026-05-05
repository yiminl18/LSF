def rule_text_span_debt_excluding_current_maturities(doc: dict) -> list[dict]:
    """Match text or table spans using the phrase debt excluding current maturities."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if "debt excluding current maturities" in txt:
                out.append(span)
        return out
    except Exception:
        return []
