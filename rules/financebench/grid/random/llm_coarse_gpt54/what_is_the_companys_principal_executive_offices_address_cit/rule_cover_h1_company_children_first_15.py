def rule_cover_h1_company_children_first_15(doc: dict) -> list[dict]:
    """Match early child spans under the company H1 on page 1, where the address usually appears."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            depth = ((span.get("structure") or {}).get("depth"))
            if path and "|" not in path and depth == 2:
                out.append(span)
        return out[:15]
    except Exception:
        return []
