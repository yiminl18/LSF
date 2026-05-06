def rule_page1_first_company_h1_after_commission_file_number(doc: dict) -> list[dict]:
    """Match the first page-1 H1 company header appearing after a commission file number mention."""
    try:
        texts = doc.get("texts", [])
        seen_file = False
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text", "") or "") + " " + (span.get("text_span", "") or "")).lower()
            if "commission file" in txt or "commission file number" in txt:
                seen_file = True
            if (
                seen_file
                and span.get("label") == "section_header"
                and span.get("structure", {}).get("level") == "H1"
                and "exact name of registrant" not in txt
                and "form 10-" not in txt
                and "form 8-k" not in txt
                and "current report" not in txt
                and "securities and exchange commission" not in txt
            ):
                out.append(span)
                break
        return out
    except Exception:
        return []
