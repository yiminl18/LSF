def rule_page1_symbol_exchange_after_headers(doc: dict) -> list[dict]:
    """Match spans immediately after 'Trading Symbol' or exchange header spans on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "trading symbol" in txt
                or "trading symbol(s)" in txt
                or "name of each exchange on which registered" in txt
                or "name of each exchange" in txt
            ):
                for j in range(i + 1, min(len(texts), i + 4)):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
