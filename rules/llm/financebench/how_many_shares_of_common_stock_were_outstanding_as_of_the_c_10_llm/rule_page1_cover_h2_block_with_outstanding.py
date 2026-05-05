def rule_page1_cover_h2_block_with_outstanding(doc: dict) -> list[dict]:
    """Match page-1 H2/H3 cover blocks that inline the outstanding-share disclosure."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            level = ((span.get("structure") or {}).get("level") or "")
            if level not in {"H2", "H3", "H4"}:
                continue
            full = ((span.get("text") or "") + " " + (span.get("text_span") or "")).lower()
            if "outstanding" in full and ("common stock" in full or "shares" in full):
                out.append(span)
    except Exception:
        return []
    return out
