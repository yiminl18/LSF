def rule_page1_amazon_item1_principal_corporate_offices(doc: dict) -> list[dict]:
    """Match Amazon Item 1 sentence stating principal corporate offices are located in Seattle, Washington."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            path = ((span.get("structure") or {}).get("path_text") or "")
            if re.search(r'Item 1\. Business|ITEM 1\. BUSINESS', path, re.I) and re.search(r'principal corporate offices are located in Seattle,\s*Washington', txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
