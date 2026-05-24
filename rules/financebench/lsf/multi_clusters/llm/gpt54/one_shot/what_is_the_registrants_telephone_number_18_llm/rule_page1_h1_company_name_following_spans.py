def rule_page1_h1_company_name_following_spans(doc: dict) -> list[dict]:
    """Match spans immediately following the company-name H1 on page 1, where phone details often appear."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                text = span.get("text", "") or ""
                if not re.search(r"form\s+10-|current report|securities and exchange commission", text, re.I):
                    for j in range(i + 1, min(len(texts), i + 20)):
                        s = texts[j]
                        if s.get("page_no") != 1:
                            break
                        blob = (s.get("text", "") or "") + " " + (s.get("text_span", "") or "")
                        if re.search(r"telephone number|area code", blob, re.I) or re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                            out.append(s)
        return out
    except Exception:
        return []
