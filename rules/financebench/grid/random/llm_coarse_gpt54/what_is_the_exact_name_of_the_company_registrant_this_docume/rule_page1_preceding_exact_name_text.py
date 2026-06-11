def rule_page1_preceding_exact_name_text(doc: dict) -> list[dict]:
    """Match the span immediately preceding a page-1 text span containing exact name of registrant."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and "exact name of registrant" in (span.get("text") or "").lower()
                and i > 0
            ):
                out.append(texts[i - 1])
        return out
    except Exception:
        return []
