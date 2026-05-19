def rule_page1_under_company_header(doc: dict) -> list[dict]:
    """Match page 1 body spans under the top company header, where cover-page registration data usually appears."""
    try:
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "")
            level = (span.get("structure", {}) or {}).get("level", "")
            if level == "Body" and path and "form 10-k" not in path.lower():
                out.append(span)
        return out
    except Exception:
        return []
