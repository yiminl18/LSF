def rule_page1_company_block_near_ein_and_zip(doc: dict) -> list[dict]:
    """Match page-1 spans in the company-identification block near EIN/zip/address markers."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if re.search(r'I\.?R\.?S\.? Employer Identification No\.?', txt, re.I) or re.search(r'\(Zip Code\)', txt, re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 3)):
                    out.append(texts[j])
        return out
    except Exception:
        return []
