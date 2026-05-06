def rule_page1_large_bold_name_before_exact_caption(doc: dict) -> list[dict]:
    """Match large bold page-1 spans immediately followed by the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            txt = (span.get("text") or "").strip()
            nxt_txt = (nxt.get("text") or "").strip().lower()
            if span.get("page_no") != 1:
                continue
            if nxt.get("page_no") != 1:
                continue
            if "exact name of registrant" not in nxt_txt:
                continue
            if span.get("bold") != 1:
                continue
            if (span.get("size") or 0) < 8:
                continue
            if any(ch.isalpha() for ch in txt):
                out.append(span)
        return out
    except Exception:
        return []
