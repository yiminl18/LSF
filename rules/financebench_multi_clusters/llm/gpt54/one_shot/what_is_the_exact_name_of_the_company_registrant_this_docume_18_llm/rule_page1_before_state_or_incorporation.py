def rule_page1_before_state_or_incorporation(doc: dict) -> list[dict]:
    """Match the span immediately preceding a page-1 span mentioning state/jurisdiction of incorporation."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(1, len(texts)):
            span = texts[i]
            txt = span.get("text") or ""
            if span.get("page_no") == 1 and (
                "State or other jurisdiction of incorporation" in txt
                or "State or other jurisdiction of incorporation or organization" in txt
            ):
                prev = texts[i - 1]
                if prev.get("page_no") == 1:
                    out.append(prev)
        return out
    except Exception:
        return []
