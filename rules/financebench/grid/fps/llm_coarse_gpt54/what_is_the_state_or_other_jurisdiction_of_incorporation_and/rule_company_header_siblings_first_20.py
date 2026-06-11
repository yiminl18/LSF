def rule_company_header_siblings_first_20(doc: dict) -> list[dict]:
    """Match early page-1 spans near the company header where state and EIN usually appear."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        company_idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if span.get("label") == "section_header" and txt:
                if any(k in txt.lower() for k in ["inc.", "incorporated", "plc", "corporation", "company"]):
                    company_idx = i
                    break
        if company_idx is None:
            return []
        for span in texts[company_idx:company_idx + 20]:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r"state\s+or\s+other\s+jurisdiction\s+of\s+incorporation", txt, re.I):
                out.append(span)
            elif re.search(r"(i\.?r\.?s\.?\s+)?employer\s+identification", txt, re.I):
                out.append(span)
            elif re.fullmatch(r"\d{2}-\d{7}", (span.get("text") or "").strip()):
                out.append(span)
        return out
    except Exception:
        return []
