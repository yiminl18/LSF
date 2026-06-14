def rule_page1_text_before_dear(doc: dict) -> list[dict]:
    """Return page-1 text spans immediately preceding a Dear salutation."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            if span.get("page_no") != 1 or span.get("label") != "text":
                continue
            nxt = texts[i + 1]
            if nxt.get("page_no") != 1:
                continue
            nxt_text = (nxt.get("text") or "").strip().lower()
            if nxt_text.startswith("dear "):
                out.append(span)
        return out
    except Exception:
        return []
