def rule_tables_with_total_assets_and_not_toc_page(doc: dict) -> list[dict]:
    """Match total-assets tables not under table-of-contents/index sections."""
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("label") != "table":
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "table of contents" in path or "index" in path:
                continue
            if "total assets" in (span.get("text") or "").lower():
                out.append(span)
        return out
    except Exception:
        return []
