def rule_cover_page_address_block(doc: dict) -> list[dict]:
    """Retrieve page-1 cover-page company block spans containing principal office address/zip cues."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            if span.get("page_no") != 1:
                continue
            label = span.get("label", "")
            path = ((span.get("structure") or {}).get("path_text") or "")
            level = ((span.get("structure") or {}).get("level") or "")
            txt = (span.get("text") or "")
            txt_span = span.get("text_span") or ""
            blob = (txt + " " + txt_span).lower()
            # Strong direct cues
            if any(k in blob for k in [
                'address of principal executive offices',
                'address and telephone number, including area code, of registrant',
                'address of principal executive offices and zip code',
                '(zip code)',
                'zip code'
            ]):
                out.append(span)
                continue
            # Top-level company block on page 1 with address-like content
            if path and 'form 10-' not in path.lower() and level in ('H1', 'H2', 'Body'):
                if re.search(r'\b\d{3,5}\b', txt) and (',' in txt or re.search(r'\b(?:street|st\.?|avenue|ave\.?|boulevard|blvd\.?|drive|dr\.?|plaza|road|rd\.?|center|hamilton|terry|park|lake|olympic|tower)\b', blob)):
                    out.append(span)
                    continue
                if re.search(r'\b\d{5}(?:-\d{4})?\b', blob) and path.count('|') <= 1:
                    out.append(span)
                    continue
        return out
    except Exception:
        return []

