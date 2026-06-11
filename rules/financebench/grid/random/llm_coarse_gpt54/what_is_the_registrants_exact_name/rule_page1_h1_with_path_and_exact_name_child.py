def rule_page1_h1_with_path_and_exact_name_child(doc: dict) -> list[dict]:
    """Match H1 headers whose path_text is reused by child spans including the exact-name caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                path = span.get("structure", {}).get("path_text") or ""
                children = [
                    s for s in texts
                    if s.get("page_no") == 1
                    and (s.get("structure", {}).get("path_text") or "") == path
                    and s is not span
                ]
                if any("exact name of registrant as specified in its charter" in ((c.get("text") or "").lower()) for c in children):
                    out.append(span)
        return out
    except Exception:
        return []
