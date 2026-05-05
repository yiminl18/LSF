def rule_page1_h1_with_following_small_caption(doc: dict) -> list[dict]:
    """Match page-1 H1 spans followed by a smaller-font caption line, typical of registrant name blocks."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            if (
                span.get("page_no") == 1
                and nxt.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and float(nxt.get("size", 0) or 0) < float(span.get("size", 0) or 0)
                and "form 10-" not in (span.get("text", "") or "").lower()
                and "form 8-k" not in (span.get("text", "") or "").lower()
                and "securities and exchange commission" not in (span.get("text", "") or "").lower()
            ):
                out.append(span)
        return out
    except Exception:
        return []
