def rule_cover_address_block_components_page1(doc: dict) -> list[dict]:
    """Retrieve page-1 registrant cover block components around the principal executive office address."""
    try:
        out = []
        street_terms = [
            'drive', 'plaza', 'avenue', 'road', 'street', 'boulevard', 'lane',
            'way', 'place', 'riverfront', 'bowerman', 'hamilton', 'rockledge',
            'tower', 'terry', 'lake', 'riverside', 'johnson'
        ]
        city_terms = [
            'beaverton', 'bethesda', 'new brunswick', 'corning', 'seattle',
            'san jose', 'issaquah', 'chicago', 'bristol', 'warmley'
        ]
        for span in doc.get('texts', []):
            if span.get('page_no') != 1:
                continue
            if span.get('label') not in {'text', 'section_header'}:
                continue
            text = (span.get('text') or '')
            low = text.lower()
            path = ((span.get('structure') or {}).get('path_text') or '').lower()
            if not path or ('form 10-k' not in path and 'inc.' not in path and 'corporation' not in path and 'plc' not in path and 'company' not in path):
                continue
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
            if any(term in low for term in street_terms) and (any(c in low for c in city_terms) or any(ch.isdigit() for ch in text)):
                out.append(span); continue
            if any(c in low for c in city_terms) and any(ch.isdigit() for ch in text):
                out.append(span); continue
            if low.strip().isdigit() and len(low.strip()) in {5, 10}:
                out.append(span); continue
        return out
    except Exception:
        return []

