def rule_company_header_with_form_context(doc: dict) -> list[dict]:
    """Match company-name headers on page 1 when they appear in immediate form/report context."""
    import re
    try:
        texts = doc.get("texts", [])
        has_form_signal = any(
            span.get("page_no") == 1 and re.search(r"(FORM\s+10-K|FORM\s+10-Q|FORM\s+8-K|CURRENT REPORT|NEWS RELEASE)", (span.get("text") or ""), re.I)
            for span in texts
        )
        if not has_form_signal:
            return []
        out = []
        for span in texts:
            txt = (span.get("text") or "").strip()
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and txt
                and not re.search(r"^(FORM|CURRENT REPORT|NEWS RELEASE|UNITED STATES|SECURITIES AND EXCHANGE COMMISSION|WASHINGTON|OR)$", txt, re.I)
            ):
                out.append(span)
        return out
    except Exception:
        return []
