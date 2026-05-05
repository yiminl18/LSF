def rule_page1_body_under_company_header_phone(doc: dict) -> list[dict]:
    """Match body spans under the main company H1 on page 1 that contain a phone number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("structure", {}).get("depth", 99) < 2:
                continue
            path = span.get("structure", {}).get("path_text", "") or ""
            text = span.get("text", "") or ""
            if path and not re.search(r"form 10-k|form 10-q|form 8-k|current report", path, re.I):
                if re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", text):
                    out.append(span)
        return out
    except Exception:
        return []
