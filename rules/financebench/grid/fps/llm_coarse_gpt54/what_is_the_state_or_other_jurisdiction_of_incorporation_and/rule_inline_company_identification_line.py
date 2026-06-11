def rule_inline_company_identification_line(doc: dict) -> list[dict]:
    """Match page-1 spans that inline the company name, state, and EIN in one long cover-page line."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r"exact name of registrant", txt, re.I) and re.search(r"\d{2}-\d{7}", txt):
                out.append(span)
        return out
    except Exception:
        return []
