def rule_page1_h1_with_descendant_commission_and_ein_cover(doc: dict) -> list[dict]:
    """Match page-1 H1 spans in cover sections where nearby descendants include EIN and exchange-registration details."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if not (span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1"):
                continue
            path = span.get("structure", {}).get("path_text") or ""
            score = 0
            for s in texts:
                spath = s.get("structure", {}).get("path_text") or ""
                txt = (s.get("text") or "").lower()
                if spath == path or spath.startswith(path + " |"):
                    if "employer identification" in txt:
                        score += 1
                    if "section 12(b)" in txt:
                        score += 1
                    if "telephone number" in txt:
                        score += 1
            if score >= 2:
                out.append(span)
        return out
    except Exception:
        return []
