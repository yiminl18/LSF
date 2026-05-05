def rule_path_company_top_page1(doc: dict) -> list[dict]:
    """Match page-1 spans under the top-level company H1 where cover-page identifiers usually appear."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            if span.get("page_no") == 1 and path and "FORM 10-" not in path.upper():
                if span.get("structure", {}).get("depth", 0) in {2, 3}:
                    out.append(span)
        return out
    except Exception:
        return []
