def rule_page1_body_spans_after_company_h1(doc: dict) -> list[dict]:
    """Match body spans immediately following the company H1 on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        company_idx = None
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in (span.get("text", "") or "")
                and "SECURITIES AND EXCHANGE COMMISSION" not in (span.get("text", "") or "")
            ):
                company_idx = i
                break
        if company_idx is None:
            return []
        for span in texts[company_idx+1:company_idx+15]:
            if span.get("page_no") == 1 and span.get("label") in {"text", "section_header"}:
                out.append(span)
        return out
    except Exception:
        return []
