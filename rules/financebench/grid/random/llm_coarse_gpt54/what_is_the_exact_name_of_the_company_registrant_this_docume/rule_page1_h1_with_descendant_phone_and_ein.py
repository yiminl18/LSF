def rule_page1_h1_with_descendant_phone_and_ein(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose descendants include both phone and EIN labels."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            path = span.get("structure", {}).get("path_text") or ""
            has_phone = False
            has_ein = False
            for s in texts:
                spath = s.get("structure", {}).get("path_text") or ""
                txt = (s.get("text") or "").lower()
                if spath == path or spath.startswith(path + " |"):
                    if "employer identification" in txt or "i.r.s. employer identification no." in txt:
                        has_ein = True
                    if "telephone number" in txt:
                        has_phone = True
            if has_phone and has_ein:
                out.append(span)
        return out
    except Exception:
        return []
