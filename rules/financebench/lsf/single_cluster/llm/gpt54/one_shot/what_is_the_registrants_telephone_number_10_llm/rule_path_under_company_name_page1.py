def rule_path_under_company_name_page1(doc: dict) -> list[dict]:
    """Match page-1 spans under the company-name path that mention telephone or contain a phone number."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s\-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b)")
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure", {}) or {}).get("path_text") or "").strip()
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if path and path not in {"FORM 10-K", "UNITED STATES SECURITIES AND EXCHANGE COMMISSION"}:
                if re.search(r"telephone|area code|registrant", text, re.I) or phone_re.search(text):
                    out.append(span)
        return out
    except Exception:
        return []
