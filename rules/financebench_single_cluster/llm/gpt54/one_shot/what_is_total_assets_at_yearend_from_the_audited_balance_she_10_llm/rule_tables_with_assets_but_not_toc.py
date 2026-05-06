def rule_tables_with_assets_but_not_toc(doc: dict) -> list[dict]:
    """Match asset-related tables excluding table-of-contents tables."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            txt = (span.get("text") or "").lower()
            if "table of contents" in path or path.endswith("| index") or "index to the form 10-k" in path:
                continue
            if "total assets" in txt or ("assets" in txt and "liabilities" in txt):
                out.append(span)
        return out
    except Exception:
        return []
