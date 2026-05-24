def rule_page1_10k_cover_phone_after_company_name(doc: dict) -> list[dict]:
    """Match phone-bearing spans after the company name on 10-K cover pages."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        company_idx = None
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("structure", {}).get("level") == "H1":
                t = span.get("text", "") or ""
                if not re.search(r"form 10-|current report|commission", t, re.I):
                    company_idx = i
                    break
        if company_idx is None:
            return []
        for span in texts[company_idx: min(len(texts), company_idx + 20)]:
            if span.get("page_no") != 1:
                break
            blob = (span.get("text", "") or "") + " " + (span.get("text_span", "") or "")
            if re.search(r"telephone number|area code", blob, re.I) or re.search(r"(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                out.append(span)
        return out
    except Exception:
        return []
