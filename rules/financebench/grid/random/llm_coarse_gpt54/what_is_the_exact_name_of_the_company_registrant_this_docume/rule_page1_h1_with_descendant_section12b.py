def rule_page1_h1_with_descendant_section12b(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose subtree contains section 12(b) registration language."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            path = span.get("structure", {}).get("path_text") or ""
            if any(
                ((s.get("structure", {}).get("path_text") or "") == path or (s.get("structure", {}).get("path_text") or "").startswith(path + " |"))
                and "section 12(b)" in (s.get("text") or "").lower()
                for s in texts
            ):
                out.append(span)
        return out
    except Exception:
        return []
