def rule_page1_company_name_not_numeric_not_label(doc: dict) -> list[dict]:
    """Match prominent page-1 nonnumeric spans that are not labels like address, zip, or commission file."""
    try:
        texts = doc.get("texts", [])
        out = []
        bad_terms = [
            "commission file", "i.r.s.", "irs", "zip code", "address of principal executive offices",
            "registrant's telephone", "registrant’s telephone", "state or other jurisdiction",
            "title of each class", "trading symbol", "name of each exchange"
        ]
        for span in texts:
            txt = (span.get("text") or "").strip()
            low = txt.lower()
            if span.get("page_no") != 1:
                continue
            if span.get("bold") != 1 or float(span.get("size") or 0) < 10:
                continue
            if any(b in low for b in bad_terms):
                continue
            if txt.replace("-", "").replace(".", "").isdigit():
                continue
            if "form 10-" in low or "form 8-k" in low or "securities and exchange commission" in low:
                continue
            out.append(span)
        return out
    except Exception:
        return []
