def rule_page1_8k_company_header_children(doc: dict) -> list[dict]:
    """Match page-1 child spans under company header in 8-K style covers where address is inline."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            depth = ((span.get("structure") or {}).get("depth"))
            if depth == 2 and path and "|" not in path:
                out.append(span)
        return out
    except Exception:
        return []
