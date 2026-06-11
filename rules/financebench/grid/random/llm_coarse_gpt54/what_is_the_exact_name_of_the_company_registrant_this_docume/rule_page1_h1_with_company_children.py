def rule_page1_h1_with_company_children(doc: dict) -> list[dict]:
    """Match page-1 H1 headers whose child/body spans include state, EIN, address, or exchange-registration boilerplate."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if not (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
            ):
                continue
            path = span.get("structure", {}).get("path_text") or ""
            child_hits = 0
            for s in texts:
                if s.get("page_no") != 1:
                    continue
                spath = s.get("structure", {}).get("path_text") or ""
                txt = (s.get("text") or "").lower()
                if spath == path or spath.startswith(path + " |"):
                    if (
                        "state or other jurisdiction" in txt
                        or "employer identification" in txt
                        or "address of principal executive offices" in txt
                        or "securities registered pursuant to section 12(b)" in txt
                        or "trading symbol" in txt
                    ):
                        child_hits += 1
            if child_hits >= 2:
                out.append(span)
        return out
    except Exception:
        return []
