def rule_tables_on_ffo1_pages_from_contents(doc: dict) -> list[dict]:
    """Match tables on pages listed in contents for FFO-1 / Summary of Fiscal Operations."""
    import re
    out = []
    try:
        texts = doc.get("texts", [])
        target_pages = set()
        for span in texts:
            txt = span.get("text", "") or ""
            if span.get("label") not in {"text", "table"}:
                continue
            m = re.search(r'FFO[\-–— ]?1\.?.{0,80}?Summary of Fiscal Operations.*?(\d{1,3})', txt, re.I | re.S)
            if m:
                try:
                    target_pages.add(int(m.group(1)))
                except Exception:
                    pass
        if not target_pages:
            return []
        for span in texts:
            if span.get("label") == "table" and span.get("page_no") in target_pages:
                out.append(span)
    except Exception:
        return []
    return out
