def rule_page1_bold_span_followed_by_exact_name_caption(doc: dict) -> list[dict]:
    """Match any bold page-1 span whose next one or two spans contain the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1 or span.get("bold") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if not any(ch.isalpha() for ch in txt):
                continue
            window = texts[i + 1:i + 3]
            if any("exact name of registrant" in ((w.get("text") or "").lower()) for w in window):
                out.append(span)
        return out
    except Exception:
        return []
