def rule_cover_principal_offices_page1(doc: dict) -> list[dict]:
    """Retrieve page-1 cover spans containing the principal executive office address or its immediate label."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "")
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            low = text.lower()
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if "address of principal executive offices" in low or "address of principal executive offices" in path:
                out.append(span)
                continue
            if "address and telephone number" in low and "principal executive offices" in low:
                out.append(span)
                continue
            if "address of principal executive offices and zip code" in low:
                out.append(span)
                continue
            if "zip code" in low and ("principal executive offices" in low or "address of principal executive offices" in path):
                out.append(span)
                continue
            if re.search(r'\b\d{1,5}\s+[A-Za-z0-9.&\- ]{3,},\s*[A-Za-z .\-]+\s+\d{5}(?:-\d{4})?\b', text):
                out.append(span)
                continue
            if re.search(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+[A-Za-z0-9.&\- ]+,\s*[A-Za-z .\-]+\s+\d{5}(?:-\d{4})?\b', low):
                out.append(span)
                continue
        return out
    except Exception:
        return []

