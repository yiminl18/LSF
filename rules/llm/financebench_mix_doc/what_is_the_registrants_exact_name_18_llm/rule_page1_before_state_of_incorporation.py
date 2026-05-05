def rule_page1_before_state_of_incorporation(doc: dict) -> list[dict]:
    """Match the span immediately before a state-of-incorporation caption on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and "state or other jurisdiction of incorporation" in (span.get("text", "") or "").lower()
                and i > 0
            ):
                prev = texts[i - 1]
                if prev.get("page_no") == 1:
                    out.append(prev)
        return out
    except Exception:
        return []
