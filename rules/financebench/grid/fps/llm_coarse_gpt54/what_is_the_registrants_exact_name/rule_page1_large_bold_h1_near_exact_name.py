def rule_page1_large_bold_h1_near_exact_name(doc: dict) -> list[dict]:
    """Match page-1 large bold H1/section_header spans that are followed shortly by an exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1:
                continue
            if span.get("label") not in {"section_header", "text"}:
                continue
            if span.get("structure", {}).get("level") not in {"H1", "Body"}:
                continue
            if float(span.get("size") or 0) < 12:
                continue
            window = texts[i + 1:i + 6]
            if any("exact name of registrant" in ((s.get("text") or "").lower()) for s in window):
                out.append(span)
        return out
    except Exception:
        return []
