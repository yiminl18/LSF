def rule_cover_path_text_company_page1(doc: dict) -> list[dict]:
    """Match page-1 body/text spans under the company cover block that mention outstanding shares."""
    try:
        out = []
        for span in doc.get("texts", []):
            struct = span.get("structure", {}) or {}
            path = (struct.get("path_text") or "").lower()
            text = (span.get("text") or "").lower()
            if span.get("page_no") in (1, 2):
                if path and "form 10-k" not in path and ("outstanding" in text or "common stock" in text):
                    out.append(span)
        return out
    except Exception:
        return []
