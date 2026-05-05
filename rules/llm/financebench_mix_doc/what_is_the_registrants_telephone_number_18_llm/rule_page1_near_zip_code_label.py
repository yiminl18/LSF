def rule_page1_near_zip_code_label(doc: dict) -> list[dict]:
    """Match page-1 spans near a zip-code label, where the phone often immediately follows."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"\(zip code\)|zip code", text, re.I):
                for j in range(i, min(len(texts), i + 5)):
                    s = texts[j]
                    blob = (s.get("text", "") or "") + " " + (s.get("text_span", "") or "")
                    if s.get("page_no") == 1 and re.search(r"(telephone number|area code)", blob, re.I):
                        out.append(s)
                    elif s.get("page_no") == 1 and re.search(r"^\s*(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}\s*$", (s.get("text", "") or "").strip()):
                        out.append(s)
        return out
    except Exception:
        return []
