def rule_page1_path_text_contains_company_and_span_has_incorporation(doc: dict) -> list[dict]:
    """Match page-1 spans under a company path_text that contain incorporation/jurisdiction wording."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}).get("path_text") or "").strip()
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and path and "commission" not in path.lower() and "form 10-k" not in path.lower():
                if re.search(r"state( or other jurisdiction)? of incorporation|state of incorporation|jurisdiction of incorporation|incorporation or organization", txt, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
