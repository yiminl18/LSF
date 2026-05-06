def rule_page1_path_company_name_and_outstanding(doc: dict) -> list[dict]:
    """Match page-1 spans under the company cover-page path that mention outstanding shares."""
    out = []
    try:
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if path and "form 10-k" not in path and ("outstanding" in text) and ("common stock" in text or "shares" in text):
                out.append(span)
    except Exception:
        return []
    return out
