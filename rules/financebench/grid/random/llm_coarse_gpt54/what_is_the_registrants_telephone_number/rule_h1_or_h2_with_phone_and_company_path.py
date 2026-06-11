def rule_h1_or_h2_with_phone_and_company_path(doc: dict) -> list[dict]:
    """Match heading spans with phone numbers under a company-name path on the first page."""
    try:
        import re
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s?\d[\d\s\-]{5,}|\(\d{3}\)\s?\d{3}[-\s]?\d{4}|\d{3}[-/]\d{3}[-/]\d{4})")
        for span in doc.get("texts", []):
            lvl = (span.get("structure", {}) or {}).get("level")
            path = (span.get("structure", {}) or {}).get("path_text", "") or ""
            if span.get("page_no") == 1 and lvl in {"H1", "H2", "H3"} and phone_re.search(span.get("text", "") or ""):
                if path and not re.search(r"FORM 10-|FORM 8-|PART I|TABLE OF CONTENTS|INDEX", path, re.I):
                    out.append(span)
        return out
    except Exception:
        return []
