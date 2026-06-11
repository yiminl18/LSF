def rule_page1_after_company_name_first_15(doc: dict) -> list[dict]:
    """Match the first 15 body spans after the company-name header on page 1 for high-recall address capture."""
    try:
        texts = doc.get("texts", [])
        out = []
        start = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = span.get("text") or ""
                if txt.isupper() or "INC." in txt or "CORPORATION" in txt or "PLC" in txt or "COMPANY" in txt:
                    if "FORM 10" not in txt and "SECURITIES AND EXCHANGE COMMISSION" not in txt:
                        start = i
                        break
        if start is None:
            return []
        for span in texts[start + 1:start + 16]:
            if span.get("page_no") == 1:
                out.append(span)
        return out
    except Exception:
        return []
