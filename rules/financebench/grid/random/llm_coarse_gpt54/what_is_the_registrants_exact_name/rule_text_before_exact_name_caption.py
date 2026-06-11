def rule_text_before_exact_name_caption(doc: dict) -> list[dict]:
    """Match any text or section_header span immediately before the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if "exact name of registrant as specified in its charter" in ((span.get("text") or "").lower()):
                if i > 0 and texts[i - 1].get("page_no") == span.get("page_no"):
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
