def rule_path_contains_company_name_page1(doc: dict) -> list[dict]:
    """Match page-1 body spans under the main company path that contain phone-like text."""
    import re
    try:
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "")
            if span.get("page_no") == 1 and path and phone_re.search(text):
                if path.strip() and "form 10-k" not in path.lower():
                    out.append(span)
        return out
    except Exception:
        return []
