def rule_page1_near_registrant_name_block(doc: dict) -> list[dict]:
    """Match early page-1 spans under the company-name H1 block that look like address content."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = ((span.get("structure") or {}).get("path_text") or "")
            txt = (span.get("text") or "").strip()
            if path and "FORM 10-" not in path and re.search(r'\d{2,}.*(?:[A-Z]{2}\s+\d{5}|California \d{5}|Washington \d{5}|United Kingdom|Bristol|New York, New York \d{5})', txt):
                out.append(span)
        return out
    except Exception:
        return []
