def rule_cover_page_registration_sequence(doc: dict) -> list[dict]:
    """Match spans in the common cover-page sequence: class -> symbol -> exchange."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "title of each class" in txt or "class b common stock" in txt or "common stock" in txt:
                window = texts[max(0, i-1):min(len(texts), i+8)]
                joined = " ".join(((s.get("text") or "") + " " + (s.get("text_span") or "")) for s in window).lower()
                if "exchange" in joined or "trading symbol" in joined or "nasdaq" in joined or "new york stock exchange" in joined:
                    out.extend(window)
        return out
    except Exception:
        return []
