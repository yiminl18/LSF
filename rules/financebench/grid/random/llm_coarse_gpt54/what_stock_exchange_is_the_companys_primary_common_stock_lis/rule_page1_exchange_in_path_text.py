def rule_page1_exchange_in_path_text(doc: dict) -> list[dict]:
    """Match spans whose path_text itself contains exchange-related wording."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            low = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "exchange on which registered" in low or "stock exchange" in low or "nasdaq" in low:
                out.append(span)
        return out
    except Exception:
        return []
