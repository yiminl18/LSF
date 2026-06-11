def rule_page1_company_name_after_commission_file_number(doc: dict) -> list[dict]:
    """Match the first prominent page-1 span after a commission file number mention."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen = False
        for span in texts:
            low = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "commission file number" in low or "commission file no." in low or "commission file no" in low or "file number" in low:
                seen = True
                continue
            if seen and span.get("bold") == 1 and float(span.get("size") or 0) >= 10:
                if "exact name of registrant" not in low and "state or other jurisdiction" not in low:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
