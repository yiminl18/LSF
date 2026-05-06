def rule_front_page_principal_office_block(doc: dict) -> list[dict]:
    """Retrieve front-page company block spans containing principal office address/location."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            text = (span.get("text") or "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            level = ((span.get("structure") or {}).get("level") or "")
            label = span.get("label") or ""
            page_no = span.get("page_no")
            bold = span.get("bold", 0)
            if page_no != 1:
                continue
            if level not in {"Body", "H2", "H1"}:
                continue
            if label not in {"text", "section_header", "table", "checkbox_selected", "checkbox_unselected"}:
                continue
            t = text.lower()
            p = path.lower()
            if "address of principal executive offices" in t or "address and telephone number, including area code, of registrant's principal executive offices" in t:
                out.append(span)
                continue
            if any(k in t for k in ["park avenue", "terry avenue north", "lake drive", "3m center", "west 34th street", "olympic boulevard", "ocean park boulevard", "tower road north", "hamilton avenue"]):
                out.append(span)
                continue
            if bold == 1 and page_no == 1 and path and path.count("|") <= 1:
                if re.search(r"\b(address|principal executive offices|zip code|telephone number|commission file|i\.r\.s\.|employer identification)\b", t):
                    out.append(span)
                    continue
            if path and path.count("|") <= 1 and bold == 1:
                if re.search(r"\b\d{2,5}\b", text) and ("," in text or "united kingdom" in t):
                    out.append(span)
                    continue
        return out
    except Exception:
        return []

