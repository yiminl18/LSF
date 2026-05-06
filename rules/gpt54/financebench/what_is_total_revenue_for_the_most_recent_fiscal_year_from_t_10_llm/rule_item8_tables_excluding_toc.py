def rule_item8_tables_excluding_toc(doc: dict) -> list[dict]:
    """Match Item 8 tables while excluding table-of-contents/index tables."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure", {}) or {}).get("path_text", "") or "").lower()
            txt = (span.get("text") or "").lower()
            if "table of contents" in path or "index" in path or "table of contents" in txt:
                continue
            if "item 8" in path or "financial statements" in path or "supplementary data" in path:
                out.append(span)
        return out
    except Exception:
        return []
