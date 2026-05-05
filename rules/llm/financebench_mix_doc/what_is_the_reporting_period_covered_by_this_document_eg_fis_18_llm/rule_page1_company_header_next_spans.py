def rule_page1_company_header_next_spans(doc: dict) -> list[dict]:
    """Return nearby page-1 spans after the company header if they contain period cues."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip()
                if txt and txt.upper() not in {"FORM 10-K", "FORM 10-Q", "FORM 8-K", "CURRENT REPORT"}:
                    for j in range(i, min(i + 12, len(texts))):
                        s = texts[j]
                        t = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
                        if re.search(r'(fiscal year ended|quarterly period ended|date of report|annual report on form 10-k)', t):
                            out.append(s)
        return out
    except Exception:
        return []
