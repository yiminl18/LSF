def rule_page1_company_block_children_address_like(doc: dict) -> list[dict]:
    """Match page-1 body spans under company block that look like address fragments."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            if span.get("page_no") != 1:
                continue
            path = (((span.get("structure") or {}).get("path_text")) or "")
            text = (span.get("text") or "").strip()
            if re.search(r'(inc\.|corporation|company|plc|com, inc\.|blizzard)', path, re.I):
                if re.search(r'^\d{1,5}\s', text) or re.search(r'^\d{5}(?:-\d{4})?$', text) or re.search(r'^[A-Z]{2}$', text) or re.search(r'^[A-Za-z .-]+,$', text):
                    out.append(span)
        return out
    except Exception:
        return []
