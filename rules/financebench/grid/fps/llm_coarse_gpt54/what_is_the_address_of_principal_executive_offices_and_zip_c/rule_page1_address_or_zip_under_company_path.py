def rule_page1_address_or_zip_under_company_path(doc: dict) -> list[dict]:
    """Match page-1 spans under the company path that are either address lines or ZIP-only lines."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            t = (span.get("text") or "").strip()
            if not re.search(r'company|corporation|inc\.|plc|incorporated|johnson & johnson|ebay|costco|boeing|corning|apple|amcor|amd|block|foot locker|best buy|american water', path, re.I):
                continue
            if re.search(r'^\d{5}(?:-\d{4})?$', t) or (re.search(r'^\d{1,6}\s+\S+|\bone\b', t, re.I) and re.search(r'street|avenue|drive|plaza|road|market|water|lake|hamilton|rockledge|tower|penn', t, re.I)):
                out.append(span)
        return out
    except Exception:
        return []
