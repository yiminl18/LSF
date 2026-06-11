def rule_cover_page_company_header_any_form(doc: dict) -> list[dict]:
    """Match the prominent company header on page 1 across 10-K, 10-Q, 8-K forms."""
    try:
        texts = doc.get("texts", [])
        form_seen = any(
            s.get("page_no") == 1 and "FORM 10-" in (s.get("text") or "").upper()
            for s in texts
        )
        if not form_seen:
            return []
        out = []
        for span in texts:
            txt = (span.get("text") or "").upper()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in txt
                and "CURRENT REPORT" not in txt
                and "SECURITIES AND EXCHANGE COMMISSION" not in txt
            ):
                out.append(span)
        return out
    except Exception:
        return []
