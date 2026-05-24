def rule_page1_before_exact_name_text_small_caption(doc: dict) -> list[dict]:
    """Match the span immediately before the small exact-name caption on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(1, len(texts)):
            if texts[i].get("page_no") == 1 and "(Exact name of registrant as specified in its charter)" in (texts[i].get("text") or ""):
                out.append(texts[i - 1])
        return out
    except Exception:
        return []
