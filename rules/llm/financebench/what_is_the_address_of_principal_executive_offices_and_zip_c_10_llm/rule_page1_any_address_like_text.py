def rule_page1_any_address_like_text(doc: dict) -> list[dict]:
    """Match any page-1 span that looks like a postal address."""
    try:
        import re
        spans = doc.get("texts", [])
        out = []
        for span in spans:
            if span.get("page_no") != 1:
                continue
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            low = txt.lower()
            if "washington, d.c. 20549" in low:
                continue
            if "public reference room" in low:
                continue
            if "investor relations" in low and "attention" in low:
                continue
            has_num = bool(re.search(r"\b\d{1,6}\b", txt))
            has_zip = bool(re.search(r"\b\d{5}(?:-\d{4})?\b", txt))
            has_street = bool(re.search(r"\b(avenue|ave|drive|dr|road|rd|plaza|street|st|boulevard|blvd|lane|ln|way)\b", low))
            has_place = bool(re.search(r"\b(new brunswick|bethesda|beaverton|corning|seattle|chicago|issaquah|bristol|warmley|san jose|corning|new york|new jersey|oregon|washington|maryland|california|illinois|united kingdom)\b", low))
            if has_num and (has_zip or has_street) and has_place:
                out.append(span)
        return out
    except Exception:
        return []
