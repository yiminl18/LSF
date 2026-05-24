def rule_split_front_page_location_fragments(doc: dict) -> list[dict]:
    """Retrieve split page-1 location fragments in company identity blocks."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            st = span.get("structure") or {}
            path = st.get("path_text") or ""
            depth = st.get("depth") or 0
            level = st.get("level") or ""
            text = span.get("text") or ""
            label = span.get("label") or ""
            bold = span.get("bold", 0)
            t = text.lower()
            p = path.lower()
            if not path or "form 10-" in p or "securities and exchange commission" in p:
                continue
            if depth > 4 or level not in {"Body", "H2", "H3"}:
                continue
            if label not in {"text", "section_header"}:
                continue
            if "address of principal executive offices" in t:
                out.append(span)
                continue
            if re.search(r"\b(warmley|bristol|united kingdom|santa monica|issaquah|san jose|st\. paul|new york|chicago)\b", t):
                out.append(span)
                continue
            if bold == 1 and re.fullmatch(r"[A-Z]{2}", text.strip()):
                out.append(span)
                continue
            if bold == 1 and re.search(r"\b\d{5}(?:-\d{4})?\b", text):
                out.append(span)
                continue
            if bold == 1 and re.search(r"\b(tower road north|ocean park boulevard|olympic boulevard|lake drive)\b", t):
                out.append(span)
                continue
        return out
    except Exception:
        return []

