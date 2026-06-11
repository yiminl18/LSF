def rule_page1_first_company_like_after_date(doc: dict) -> list[dict]:
    """Match the first company-like prominent span after the report date line on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen_date = False
        for span in texts:
            low = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "date of report" in low or "for the fiscal year ended" in low or "for the quarterly period ended" in low:
                seen_date = True
                continue
            if seen_date and span.get("bold") == 1 and float(span.get("size") or 0) >= 10:
                if "form 10-" not in low and "form 8-k" not in low and "current report" not in low and "pursuant to section" not in low:
                    out.append(span)
                    break
        return out
    except Exception:
        return []
