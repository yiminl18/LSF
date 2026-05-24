def rule_page1_after_company_name_window(doc: dict) -> list[dict]:
    """Match a broad window of page-1 spans after the company-name header where state and EIN usually appear."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if (
                span.get("page_no") == 1
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "FORM 10-" not in (span.get("text", "") or "").upper()
                and "SECURITIES AND EXCHANGE COMMISSION" not in (span.get("text", "") or "").upper()
            ):
                for j in range(i, min(i + 20, len(texts))):
                    if texts[j].get("page_no") == 1:
                        out.append(texts[j])
        return out
    except Exception:
        return []
