def rule_exhibit_section_precise_8k_10q(doc: dict) -> list[dict]:
    """Retrieve only exhibit list/table spans in 8-K and 10-Q exhibit sections."""
    try:
        import re
        out = []
        for span in doc.get('texts', []):
            text = (span.get('text') or '')
            low = text.lower()
            label = span.get('label')
            path = ((span.get('structure') or {}).get('path_text') or '').lower()

            is_8k_10q_exhibit_area = (
                'form 8-k' in path
                or 'form 10-q' in path
                or 'item 6. exhibits' in path
                or 'item 9.01. financial statements and exhibits' in path
            )
            if not is_8k_10q_exhibit_area:
                continue

            if label == 'table':
                cells = (((span.get('table_data') or {}).get('cells')) or [])
                cell_text = ' '.join((c.get('text') or '') for c in cells).lower()
                if (
                    'exhibit no.' in cell_text
                    or 'exhibit number' in cell_text
                    or 'description' in cell_text
                    or 'cover page interactive data file' in cell_text
                ):
                    out.append(span)
                continue

            if label in {'text', 'section_header'}:
                if (
                    low.strip() == '(d) exhibits.'
                    or low.strip() == 'exhibit index'
                    or re.search(r'^\s*(?:\d+\.?\d*|104)\.\s+', text) is not None
                    or 'cover page interactive data file' in low
                    or 'press release dated' in low
                    or 'bylaws as amended' in low
                    or 'none listed' in low
                    or low.strip() in {'none', 'none.'}
                ):
                    out.append(span)
        return out
    except Exception:
        return []
