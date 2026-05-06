def rule_page1_path_text_contains_company_and_span_has_ein(doc: dict) -> list[dict]:
    """Match page-1 spans under a company path_text that contain an EIN number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").strip()
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and path and "commission" not in path.lower() and "form 10-k" not in path.lower():
                if re.search(r"\d{2}-\d{7}", txt):
                    out.append(span)
        return out
    except Exception:
        return []
