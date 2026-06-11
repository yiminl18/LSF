def rule_form_code_near_company_name_cover(doc: dict) -> list[dict]:
    """Match form-code spans occurring before the first large company-name H1 on the cover page."""
    import re
    try:
        texts = doc.get("texts", [])
        company_idx = None
        for i, span in enumerate(texts[:30]):
            if span.get("page_no") == 1 and span.get("label") == "section_header":
                txt = (span.get("text") or "").strip()
                if txt and not re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", txt, re.I) and len(txt) > 3:
                    company_idx = i
                    break
        out = []
        for i, span in enumerate(texts[:30]):
            if company_idx is not None and i > company_idx:
                break
            if re.search(r"\bFORM\s+(10-K|10-Q|8-K)\b", span.get("text") or "", re.I):
                out.append(span)
        return out
    except Exception:
        return []
