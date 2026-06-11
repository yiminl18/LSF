def rule_page1_8k_preceding_span_before_address_of_principal_executive_offices(doc: dict) -> list[dict]:
    """Return the span before '(Address of principal executive offices)' on page 1 in 8-K/10-Q layouts."""
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
