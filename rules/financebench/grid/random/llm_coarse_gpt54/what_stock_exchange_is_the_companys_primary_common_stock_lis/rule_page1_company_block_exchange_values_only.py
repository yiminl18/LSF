def rule_page1_company_block_exchange_values_only(doc: dict) -> list[dict]:
    """Match likely answer spans in the page-1 company-information block by excluding labels and keeping exchange-value text."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        company_block = False
        for span in texts:
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if span.get("label") == "section_header" and txt and "form 10-" not in low and "securities and exchange commission" not in low:
                company_block = True
            if company_block:
                if re.fullmatch(r"(?i)(the )?(new york stock exchange|nasdaq|nasdaq global select market|nasdaq global market|new york stock exchange)", txt):
                    out.append(span)
        return out
    except Exception:
        return []
