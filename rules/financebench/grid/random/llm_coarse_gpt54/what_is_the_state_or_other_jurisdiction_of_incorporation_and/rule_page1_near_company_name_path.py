def rule_page1_near_company_name_path(doc: dict) -> list[dict]:
    """Match page-1 body spans under the top company H1 where state/EIN usually appear."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            if (
                span.get("page_no") == 1
                and span.get("structure", {}).get("depth", 0) >= 2
                and path
                and "FORM 10-" not in path
                and span.get("label") in {"text", "section_header"}
            ):
                out.append(span)
        return out
    except Exception:
        return []
