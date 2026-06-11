def rule_page1_exact_name_caption_textspan_parent(doc: dict) -> list[dict]:
    """Match spans whose text is followed by text_span containing the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if "exact name of registrant" in ((span.get("text_span") or "").lower()):
                out.append(span)
        return out
    except Exception:
        return []
