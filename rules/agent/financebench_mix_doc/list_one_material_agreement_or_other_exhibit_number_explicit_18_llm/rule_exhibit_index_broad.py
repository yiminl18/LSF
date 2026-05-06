def rule_exhibit_index_broad(doc: dict) -> list[dict]:
    """Retrieve exhibit index rows/spans and nearby exhibit-related headers."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for span in texts:
            text = (span.get("text") or "")
            label = span.get("label")
            path = ((span.get("structure") or {}).get("path_text") or "")
            level = ((span.get("structure") or {}).get("level") or "")
            low = text.lower()
            path_low = path.lower()

            exhibit_path = (
                "exhibit" in path_low
                or "item 15" in path_low
                or "item 6." in path_low
                or "item 6 | exhibits" in path_low
                or "financial statements and exhibits" in path_low
                or "financial statements and schedules" in path_low
            )

            exhibit_text = (
                re.search(r'\bexhibit\s+\d+(?:\.\d+)?\b', low) is not None
                or re.search(r'\b\d+(?:\.\d+)?\s*\|', text) is not None and 'exhibit' in low
                or 'exhibit index' in low
                or low.strip().startswith('item 15') and 'exhibit' in low
            )

            if label == 'table':
                cells = (((span.get('table_data') or {}).get('cells')) or [])
                cell_text = ' '.join((c.get('text') or '') for c in cells).lower()
                if exhibit_path or 'exhibit index' in cell_text or re.search(r'\bexhibit\s+\d+(?:\.\d+)?\b', cell_text):
                    out.append(span)
                    continue

            if exhibit_path and label in {'section_header', 'text', 'list_item'}:
                if level in {'H2', 'H3', 'H4'} or exhibit_text or 'exhibit' in low:
                    out.append(span)
                    continue

            if exhibit_text and label in {'text', 'section_header', 'list_item'}:
                out.append(span)
        return out
    except Exception:
        return []
