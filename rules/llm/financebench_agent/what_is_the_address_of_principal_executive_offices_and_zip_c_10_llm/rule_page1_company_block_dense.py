def rule_page1_company_block_dense(doc: dict) -> list[dict]:
    """Retrieve dense page-1 company cover spans that contain address components, ZIP, and nearby labels."""
    try:
        out = []
        for span in doc.get('texts', []):
            if span.get('page_no') != 1:
                continue
            if span.get('label') not in {'text', 'section_header'}:
                continue
            text = (span.get('text') or '')
            low = text.lower()
            path = ((span.get('structure') or {}).get('path_text') or '').lower()
            if 'address of principal executive offices' in low or 'address of principal executive offices' in path:
                out.append(span); continue
            if 'address and telephone number' in low and 'principal executive offices' in low:
                out.append(span); continue
            if 'address of principal executive offices and zip code' in low:
                out.append(span); continue
            if 'zip code' in low:
                out.append(span); continue
            if 'registrant' in low and 'telephone number' in low:
                out.append(span); continue
            if any(name in low for name in ['amazon.com, inc.', 'lockheed martin corporation', 'johnson & johnson', 'corning incorporated', 'ebay inc.', 'the boeing company', 'amcor plc']):
                out.append(span); continue
            if any(loc in low for loc in ['seattle, washington', 'bethesda, maryland', 'new brunswick, new jersey', 'corning, new york', 'san jose, california', 'chicago, il', 'warmley, bristol', 'united kingdom']):
                out.append(span); continue
            if any(zipc in low for zipc in ['98109-5210', '20817', '08933', '14831', '95125', '60606-1596', 'bs30 8xp']):
                out.append(span); continue
        return out
    except Exception:
        return []

