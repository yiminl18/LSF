def rule_page1_texts_with_parent_h1_company_and_short_value(doc: dict) -> list[dict]:
    """Match short child text spans under a company H1 cover header, capturing state and EIN values."""
    try:
        import re
        texts = doc.get("texts", [])
        h1_company_ids = set()
        for idx, span in enumerate(texts):
            if span.get("page_no") == 1 and span.get("label") == "section_header" and span.get("structure", {}).get("level") == "H1":
                txt = (span.get("text") or "").lower()
                if txt and "commission" not in txt and "form 10-k" not in txt:
                    h1_company_ids.add(idx)
        out = []
        for span in texts:
            if span.get("page_no") == 1 and span.get("structure", {}).get("parent_id") in h1_company_ids:
                txt = (span.get("text") or "").strip()
                if len(txt.split()) <= 6 and (re.fullmatch(r"\d{2}-\d{7}", txt) or not re.search(r"address|telephone|exchange|common stock|none", txt, re.I)):
                    out.append(span)
        return out
    except Exception:
        return []
