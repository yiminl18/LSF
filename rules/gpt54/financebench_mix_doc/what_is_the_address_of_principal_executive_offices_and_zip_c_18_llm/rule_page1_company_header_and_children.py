def rule_page1_company_header_and_children(doc: dict) -> list[dict]:
    """Retrieve compact page-1 company cover spans around the registrant identity/address block."""
    try:
        texts = doc.get('texts', [])
        out = []
        for span in texts:
            if span.get('page_no') != 1:
                continue
            label = span.get('label', '')
            st = span.get('structure') or {}
            path = st.get('path_text') or ''
            level = st.get('level') or ''
            txt = span.get('text') or ''
            low = txt.lower()
            # Exclude generic filing headers and long narrative/exhibit sections
            if any(bad in low for bad in ['securities and exchange commission', 'form 10-k', 'form 10-q', 'form 8-k', 'current report']):
                continue
            if any(bad in path.lower() for bad in ['securities and exchange commission', 'form 10-k', 'form 10-q', 'form 8-k', 'current report']):
                continue
            if len(txt.split()) > 80:
                continue
            # Keep concise company header and immediate body children only
            if label == 'section_header' and level == 'H1' and path and '|' not in path:
                out.append(span)
                continue
            if level == 'Body' and path and '|' not in path:
                if any(k in low for k in [
                    'exact name of registrant', 'state or other jurisdiction', 'i.r.s. employer',
                    'address of principal executive offices', 'address of principal executive offices and zip code',
                    'address and telephone number, including area code, of registrant',
                    'zip code', 'telephone number', 'principal executive offices'
                ]):
                    out.append(span)
                    continue
                # concise address-like body span in company block
                if any(ch.isdigit() for ch in txt) and len(txt.split()) <= 20:
                    out.append(span)
                    continue
            # Some docs encode address as H2/H3 under company H1; keep only concise ones
            if label == 'section_header' and level in ('H2', 'H3') and path.count('|') <= 1 and len(txt.split()) <= 20:
                out.append(span)
                continue
        return out
    except Exception:
        return []

