def rule_page1_registered_exchange_triplet_window(doc: dict) -> list[dict]:
    """Return windows around 'Title of each class'/'Trading Symbol'/'Name of each exchange' triplets on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(len(texts) - 2):
            if any(texts[k].get("page_no") != 1 for k in (i, i+1, i+2)):
                continue
            joined = " ".join((texts[k].get("text") or "") for k in range(i, min(i+8, len(texts)))).lower()
            if "title of each class" in joined and "trading symbol" in joined and "exchange" in joined:
                for k in range(i, min(i+8, len(texts))):
                    if texts[k].get("page_no") == 1:
                        out.append(texts[k])
        return out
    except Exception:
        return []
