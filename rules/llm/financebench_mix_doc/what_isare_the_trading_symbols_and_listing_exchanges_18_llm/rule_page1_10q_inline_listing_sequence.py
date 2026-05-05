def rule_page1_10q_inline_listing_sequence(doc: dict) -> list[dict]:
    """Match 10-Q cover spans where title, symbol, and exchange appear inline or split nearby."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").lower()
            if "securities registered pursuant to section 12(b)" in txt:
                for j in range(i, min(len(texts), i + 10)):
                    s = texts[j]
                    if s.get("page_no") == 1:
                        out.append(s)
        return out
    except Exception:
        return []
