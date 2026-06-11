def rule_form_type_from_first_page_semantics(doc: dict) -> list[dict]:
    """Infer document type from first-page semantics and return supporting spans."""
    import re
    try:
        texts = [s for s in doc.get("texts", []) if s.get("page_no") == 1]
        out = []
        for span in texts:
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if re.search(r"\bFORM\s+10-K\b", txt, re.I) or re.search(r"\bANNUAL REPORT PURSUANT TO SECTION 13 OR 15\(d\)\b", txt, re.I):
                out.append(span)
            elif re.search(r"\bFORM\s+10-Q\b", txt, re.I) or re.search(r"\bQUARTERLY REPORT PURSUANT TO SECTION 13 OR 15\(d\)\b", txt, re.I):
                out.append(span)
            elif re.search(r"\bFORM\s+8-K\b", txt, re.I) or re.search(r"\bCURRENT REPORT\b", txt, re.I):
                out.append(span)
            elif re.search(r"\bNEWS RELEASE\b", txt, re.I) or re.search(r"Reports .* Financial Results", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
