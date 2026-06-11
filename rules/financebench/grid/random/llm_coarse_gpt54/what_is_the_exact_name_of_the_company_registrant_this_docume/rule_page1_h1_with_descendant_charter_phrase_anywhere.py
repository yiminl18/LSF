def rule_page1_h1_with_descendant_charter_phrase_anywhere(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose subtree contains the charter phrase anywhere in text or text_span."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            path = span.get("structure", {}).get("path_text") or ""
            found = False
            for s in texts:
                spath = s.get("structure", {}).get("path_text") or ""
                blob = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
                if (spath == path or spath.startswith(path + " |")) and "specified in its charter" in blob:
                    found = True
                    break
            if found:
                out.append(span)
        return out
    except Exception:
        return []
