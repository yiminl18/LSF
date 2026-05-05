def rule_page1_exact_name_following_text(doc: dict) -> list[dict]:
    """Match spans on page 1 immediately followed by a body span containing '(Exact name of registrant as specified in its charter)'."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            if (
                span.get("page_no") == 1
                and nxt.get("page_no") == 1
                and "(Exact name of registrant as specified in its charter)" in (nxt.get("text") or "")
            ):
                out.append(span)
        return out
    except Exception:
        return []
