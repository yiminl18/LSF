def rule_page1_text_before_exact_name_caption(doc: dict) -> list[dict]:
    """Match page-1 text spans immediately preceding the exact-name caption, for split-header layouts."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i in range(1, len(texts)):
            span = texts[i]
            prev = texts[i - 1]
            if (
                span.get("page_no") == 1
                and "exact name of registrant" in (span.get("text", "") or "").lower()
                and prev.get("page_no") == 1
                and (prev.get("label") in {"text", "section_header"})
            ):
                out.append(prev)
        return out
    except Exception:
        return []
