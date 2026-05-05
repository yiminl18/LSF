def rule_page1_parent_path_of_part_i(doc: dict) -> list[dict]:
    """Match the page-1 company root span whose text appears as the first segment of later PART I path_texts."""
    try:
        texts = doc.get("texts", [])
        roots = {}
        for span in texts:
            path = (span.get("structure", {}).get("path_text", "") or "")
            if "| PART I" in path:
                root = path.split("|")[0].strip()
                roots[root] = True
        out = []
        for span in texts:
            txt = (span.get("text", "") or "").strip()
            if span.get("page_no") == 1 and txt in roots:
                out.append(span)
        return out
    except Exception:
        return []
