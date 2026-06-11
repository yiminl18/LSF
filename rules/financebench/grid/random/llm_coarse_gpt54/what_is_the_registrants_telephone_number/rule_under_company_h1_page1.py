def rule_under_company_h1_page1(doc: dict) -> list[dict]:
    """Match page-1 spans under the main company header rather than later report sections."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if span.get("page_no") != 1:
                continue
            if not path:
                continue
            if "PART I" in path or "TABLE OF CONTENTS" in path or "INDEX" in path:
                continue
            if "FORM 10-K" in path or "FORM 10-Q" in path or "FORM 8-K" in path:
                continue
            out.append(span)
        return out
    except Exception:
        return []
