def rule_page1_symbol_like_spans_near_exchange(doc: dict) -> list[dict]:
    """Match short ticker-like page 1 spans near exchange-related text."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.fullmatch(r"[A-Z]{1,6}(?:\d+[A-Z]{0,3})?", txt):
                window = texts[max(0, i - 5): min(len(texts), i + 6)]
                joined = " ".join((w.get("text") or "") for w in window)
                if re.search(r"trading symbol|symbol|exchange|section\s+12\(b\)|registered", joined, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
