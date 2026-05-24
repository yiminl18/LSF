def rule_title_trading_exchange_triplet(doc: dict) -> list[dict]:
    """Match spans near the cover-page triplet 'Title of each class' / 'Trading Symbol' / 'Name of each exchange'."""
    try:
        texts = doc.get("texts", [])
        out = []
        trigger_idxs = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if any(k in txt for k in [
                "title of each class",
                "trading symbol",
                "trading symbol(s)",
                "name of each exchange on which registered",
                "name of each exchange on which",
                "name of each exchange",
            ]):
                trigger_idxs.append(i)
        for i in trigger_idxs:
            for j in range(max(0, i - 2), min(len(texts), i + 8)):
                out.append(texts[j])
        return out
    except Exception:
        return []
