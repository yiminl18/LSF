def rule_page1_exchange_name_or_exchange_registered(doc: dict) -> list[dict]:
    """Match page-1 spans mentioning exchange registration labels in varied wording."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "name of each exchange on which registered" in txt
                or "name of each exchange on which" in txt
                or "name of each exchange" in txt
                or "exchange on which registered" in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
