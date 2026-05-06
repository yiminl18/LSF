def rule_securities_12b_with_symbol(doc: dict) -> list[dict]:
    """Match Section 12(b) securities info AND Item 5 Market section symbol info."""
    try:
        texts = doc.get('texts', [])
        result = []
        seen = set()

        # Part 1: Section 12(b) on page 1-2
        # Capture from "Securities registered...12(b)" to "indicate by check mark"
        in_section = False
        for i, s in enumerate(texts):
            page = s.get('page_no', 0)
            if page > 2:
                continue
            text = s.get('text', '').lower()
            if 'securities registered' in text and '12(b)' in text:
                in_section = True
            if in_section:
                if id(s) not in seen:
                    result.append(s)
                    seen.add(id(s))
                # Stop at "indicate by check mark" (more reliable than 12(g))
                if 'indicate by check mark' in text:
                    break

        # Part 2: Page 1-2 exchange name spans (for documents with out-of-order spans)
        exchange_names = ['new york stock exchange', 'nasdaq', 'nyse', 'chicago stock exchange']
        for s in texts:
            page = s.get('page_no', 0)
            if page > 2:
                continue
            text = s.get('text', '').lower()
            if any(ex in text for ex in exchange_names):
                if id(s) not in seen:
                    result.append(s)
                    seen.add(id(s))

        # Part 3: Item 5 Market section spans (for older filings like Boeing, Costco 2018)
        for s in texts:
            text = s.get('text', '').lower()
            path = s.get('structure', {}).get('path_text', '').lower()
            if ('item 5' in path or 'market for registrant' in path):
                if ('trades under' in text and 'symbol' in text) or \
                   ('traded on' in text and 'symbol' in text):
                    if id(s) not in seen:
                        result.append(s)
                        seen.add(id(s))

        return result
    except Exception:
        return []
