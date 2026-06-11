def rule_press_release_page1(doc: dict) -> list[dict]:
    """Match page-1 spans explicitly saying Press Release, often attached to 8-Ks but also useful as document-type evidence."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") == 1 and "press release" in (span.get("text") or "").strip().lower():
                out.append(span)
        return out
    except Exception:
        return []
