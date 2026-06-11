def rule_page1_state_or_irs_body_text(doc: dict) -> list[dict]:
    """Match page-1 body/text spans that are the explanatory labels for state or IRS number."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            txt = (span.get("text") or "").strip()
            if re.fullmatch(r"\(?state\s+or\s+other\s+jurisdiction\s+of\s+incorporation.*\)?", txt, re.I):
                out.append(span)
            elif re.fullmatch(r"\(?i\.?r\.?s\.?\s+employer\s+identification\s+no\.?\)?", txt, re.I):
                out.append(span)
            elif re.fullmatch(r"\(?irs\s+employer\s+identification\s+no\.?\)?", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
