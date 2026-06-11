def rule_page1_preceding_span_before_address_label(doc: dict) -> list[dict]:
    """Return the span immediately preceding a page-1 '(Address of principal executive offices)' label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'address of principal executive offices', txt, re.I):
                if i > 0:
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
