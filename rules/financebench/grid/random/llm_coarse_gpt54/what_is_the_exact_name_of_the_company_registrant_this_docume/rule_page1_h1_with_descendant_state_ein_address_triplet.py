def rule_page1_h1_with_descendant_state_ein_address_triplet(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose descendants include state, EIN, and address labels together."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            path = span.get("structure", {}).get("path_text") or ""
            has_state = has_ein = has_addr = False
            for s in texts:
                spath = s.get("structure", {}).get("path_text") or ""
                txt = (s.get("text") or "").lower()
                if spath == path or spath.startswith(path + " |"):
                    has_state |= "state or other jurisdiction" in txt
                    has_ein |= "employer identification" in txt or "i.r.s. employer identification no." in txt
                    has_addr |= "address of principal executive offices" in txt
            if has_state and has_ein and has_addr:
                out.append(span)
        return out
    except Exception:
        return []
