def rule_page1_exchange_registered_keyword(doc: dict) -> list[dict]:
    """Match page-1 spans containing 'Name of each exchange on which registered' or similar exchange-registration wording."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = span.get("text", "") or ""
            if span.get("page_no") == 1 and (
                re.search(r"name of each exchange on which registered", txt, re.I)
                or re.search(r"name of exchange on which registered", txt, re.I)
                or re.search(r"name of each exchange", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
