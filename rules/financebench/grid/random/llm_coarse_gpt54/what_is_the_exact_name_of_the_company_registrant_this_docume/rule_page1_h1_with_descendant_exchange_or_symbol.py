def rule_page1_h1_with_descendant_exchange_or_symbol(doc: dict) -> list[dict]:
    """Match page-1 H1 spans whose descendants include exchange or trading symbol labels."""
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
                txt = (s.get("text") or "").lower()
                if spath == path or spath.startswith(path + " |"):
                    if "trading symbol" in txt or "name of each exchange" in txt:
                        found = True
                        break
            if found:
                out.append(span)
        return out
    except Exception:
        return []
