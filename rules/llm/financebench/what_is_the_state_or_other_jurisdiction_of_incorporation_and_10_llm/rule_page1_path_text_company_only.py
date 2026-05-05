def rule_page1_path_text_company_only(doc: dict) -> list[dict]:
    """Match page-1 spans whose path_text is just the company name block, where cover metadata usually lives."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").strip()
            if span.get("page_no") == 1 and path and "|" not in path:
                if "commission" not in path.lower() and "form 10-k" not in path.lower():
                    out.append(span)
        return out
    except Exception:
        return []
