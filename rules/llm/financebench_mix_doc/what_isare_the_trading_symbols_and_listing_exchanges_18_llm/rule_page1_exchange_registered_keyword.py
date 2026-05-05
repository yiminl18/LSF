def rule_page1_exchange_registered_keyword(doc: dict) -> list[dict]:
    """Match spans on page 1 containing exchange-registration keywords."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "name of each exchange on which registered" in txt
                or "name of each exchange" in txt
                or "exchange on which registered" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
