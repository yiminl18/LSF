def rule_page1_cover_block_before_part_i(doc: dict) -> list[dict]:
    """Match pre-Part-I page-1/2 spans mentioning outstanding shares."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            t = (span.get("text") or "").lower()
            if span.get("page_no") in (1, 2) and "part i" not in path:
                if "outstanding" in t and ("common stock" in t or "shares" in t):
                    out.append(span)
        return out
    except Exception:
        return []
