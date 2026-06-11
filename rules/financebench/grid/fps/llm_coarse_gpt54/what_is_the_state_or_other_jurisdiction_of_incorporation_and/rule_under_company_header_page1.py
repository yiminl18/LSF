def rule_under_company_header_page1(doc: dict) -> list[dict]:
    """Match page-1 spans under the company-name header that contain state/EIN cues."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if path and "FORM 10-K" not in path and "FORM 8-K" not in path and "FORM 10-Q" not in path:
                if re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", txt, re.I):
                    out.append(span)
                elif re.search(r"employer\s+identification", txt, re.I):
                    out.append(span)
                elif re.fullmatch(r"\d{2}-\d{7}", (span.get("text") or "").strip()):
                    out.append(span)
        return out
    except Exception:
        return []
