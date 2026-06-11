def rule_page1_company_name_from_path_root(doc: dict) -> list[dict]:
    """Match spans whose text equals the root section name used by many subsequent page-1 spans."""
    try:
        texts = doc.get("texts", [])
        from collections import Counter
        roots = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = span.get("structure", {}).get("path_text") or ""
            if path:
                roots.append(path.split(" | ")[0].strip())
        counts = Counter(r for r in roots if r)
        common = {r for r, c in counts.items() if c >= 2}
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and txt in common:
                out.append(span)
        return out
    except Exception:
        return []
