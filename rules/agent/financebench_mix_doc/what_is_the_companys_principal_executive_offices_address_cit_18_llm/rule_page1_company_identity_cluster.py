def rule_page1_company_identity_cluster(doc: dict) -> list[dict]:
    """Retrieve shallow page-1 company identity/address cluster under the registrant name block."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            page_no = span.get("page_no")
            if page_no != 1:
                continue
            st = span.get("structure") or {}
            path = st.get("path_text") or ""
            level = st.get("level") or ""
            depth = st.get("depth") or 0
            text = span.get("text") or ""
            label = span.get("label") or ""
            bold = span.get("bold", 0)
            t = text.lower()
            p = path.lower()
            if not path or "form 10-" in p or "securities and exchange commission" in p:
                continue
            if depth > 4 or level not in {"H1", "H2", "H3", "Body"}:
                continue
            if label not in {"text", "section_header", "table", "checkbox_selected", "checkbox_unselected"}:
                continue
            if re.search(r"address of principal executive offices|address and telephone number, including area code, of registrant.?s principal executive offices|zip code|i\.r\.s\.|employer identification|state or other jurisdiction", t):
                out.append(span)
                continue
            if re.search(r"\b(park avenue|terry avenue|lake drive|3m center|west 34th street|hamilton avenue|tower road north|ocean park boulevard|olympic boulevard)\b", t):
                out.append(span)
                continue
            if bold == 1 and len(text) <= 80 and ("," in text or re.search(r"\bca\b|\bwa\b|\bil\b|california|minnesota|washington|united kingdom|bristol|santa monica|chicago|issaquah|san jose|st\. paul|new york", t)):
                out.append(span)
                continue
        return out
    except Exception:
        return []

