def rule_page1_before_phone_number_address_candidate(doc: dict) -> list[dict]:
    """Match the span immediately before a phone-number-only span in the page-1 cover block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.fullmatch(r'[\(\+\d][\d\(\)\-\+\s]{6,}', txt):
                if i > 0:
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
