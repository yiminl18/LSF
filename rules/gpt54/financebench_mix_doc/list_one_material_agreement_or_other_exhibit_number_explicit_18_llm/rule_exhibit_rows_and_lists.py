def rule_exhibit_rows_and_lists(doc: dict) -> list[dict]:
    """Retrieve exhibit index tables and exhibit list lines under Item 6/Item 9.01 exhibit sections."""
    try:
        import re
        out = []
        for span in doc.get('texts', []):
            text = (span.get('text') or '')
            low = text.lower()
            label = span.get('label')
            path = ((span.get('structure') or {}).get('path_text') or '').lower()

            in_exhibit_section = (
                'item 6. exhibits' in path
                or 'item 6 | exhibits' in path
                or 'item 9.01. financial statements and exhibits' in path
                or 'item 15' in path
                or 'exhibit index' in path
            )

            if label == 'table':
                cells = (((span.get('table_data') or {}).get('cells')) or [])
                cell_text = ' '.join((c.get('text') or '') for c in cells).lower()
                if (
                    'exhibit index' in cell_text
                    or 'exhibit number' in cell_text
                    or 'exhibit no.' in cell_text
                    or ('description' in cell_text and re.search(r'\b(?:10|99|104|4)\.?\d*\b', cell_text))
                    or (in_exhibit_section and ('exhibit' in cell_text or 'cover page interactive data file' in cell_text))
                ):
                    out.append(span)
                    continue

            if label in {'text', 'section_header', 'list_item'} and in_exhibit_section:
                if (
                    'exhibit index' in low
                    or re.search(r'^\s*(?:\(?d\)?\s*)?exhibits?\.?$', low) is not None
                    or re.search(r'^\s*\d+(?:\.\d+)?\.\s+', text) is not None
                    or re.search(r'^\s*exhibit\s+\d+(?:\.\d+)?', low) is not None
                    or 'cover page interactive data file' in low
                    or 'press release dated' in low
                    or 'bylaws as amended' in low
                    or 'supplemental indenture' in low
                    or 'settlement agreement' in low
                    or 'credit agreement' in low
                    or 'stock incentive plan' in low
                    or 'employee stock purchase plan' in low
                    or 'equity incentive plan' in low
                    or low.strip() in {'none', 'none listed', 'none.'}
                ):
                    out.append(span)
        return out
    except Exception:
        return []
