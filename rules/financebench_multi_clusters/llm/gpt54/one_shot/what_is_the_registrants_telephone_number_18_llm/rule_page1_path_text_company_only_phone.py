def rule_page1_path_text_company_only_phone(doc: dict) -> list[dict]:
    """Match page-1 spans under a company path_text, excluding form headers, that contain phone clues."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = span.get("structure", {}).get("path_text", "") or ""
            if span.get("page_no") != 1 or not path:
                continue
            if re.search(r"form 10-|current report|securities and exchange commission", path, re.I):
                continue
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"telephone number|area code", blob, re.I) or re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
