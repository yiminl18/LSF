def rule_page1_after_company_name_until_documents_incorporated(doc: dict) -> list[dict]:
    """Match page 1 spans between company-name header and 'Documents Incorporated by Reference' containing registration info."""
    try:
        import re
        texts = doc.get("texts", [])
        start = None
        end = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "")
            if start is None and span.get("page_no") == 1 and span.get("label") == "section_header" and re.search(r"(inc\.|corporation|plc|incorporated|company)", txt, re.I):
                start = i
            if start is not None and span.get("page_no") == 1 and re.search(r"documents incorporated by reference", txt, re.I):
                end = i
                break
        if start is None:
            return []
        if end is None:
            end = min(len(texts), start + 40)
        out = []
        for s in texts[start:end]:
            t = (s.get("text") or "")
            if re.search(r"12\(b\)|trading symbol|exchange|registered|nasdaq|new york stock exchange|nyse", t, re.I) or re.fullmatch(r"[A-Z]{1,8}(?:\d+[A-Z]{0,3})?", t.strip()):
                out.append(s)
        return out
    except Exception:
        return []
