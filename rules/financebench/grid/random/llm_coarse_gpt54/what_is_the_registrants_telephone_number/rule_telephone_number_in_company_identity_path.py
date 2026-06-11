def rule_telephone_number_in_company_identity_path(doc: dict) -> list[dict]:
    """Match phone-number spans whose path_text is the company identity path on the cover page."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|[0-9]{3}[-/][0-9]{3}[-/][0-9]{4})")
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if span.get("page_no") == 1 and path and not re.search(r"FORM 10-|FORM 8-|PART I|TABLE OF CONTENTS|INDEX", path, re.I):
                if phone_re.search(span.get("text", "") or ""):
                    out.append(span)
        return out
    except Exception:
        return []
