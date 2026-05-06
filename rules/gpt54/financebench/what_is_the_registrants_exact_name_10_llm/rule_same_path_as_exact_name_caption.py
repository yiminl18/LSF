def rule_same_path_as_exact_name_caption(doc: dict) -> list[dict]:
    """Match all page-1 spans sharing the same path_text as the exact-name caption block."""
    try:
        texts = doc.get("texts", [])
        paths = set()
        for span in texts:
            txt = (span.get("text") or "").strip().lower()
            if span.get("page_no") == 1 and "exact name of registrant" in txt:
                p = span.get("structure", {}).get("path_text")
                if p:
                    paths.add(p)
        out = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("structure", {}).get("path_text") in paths:
                out.append(span)
        return out
    except Exception:
        return []
