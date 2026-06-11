def rule_page1_h1_with_descendant_exact_name_text(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose descendants include a body text exactly containing the exact-name label."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            path = span.get("structure", {}).get("path_text") or ""
            for s in texts:
                spath = s.get("structure", {}).get("path_text") or ""
                txt = (s.get("text") or "").strip().lower()
                if (spath == path or spath.startswith(path + " |")) and txt == "(exact name of registrant as specified in its charter)":
                    out.append(span)
                    break
        return out
    except Exception:
        return []
