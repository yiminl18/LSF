def rule_page1_near_exact_name_and_address(doc: dict) -> list[dict]:
    """Match page-1 spans in the cover identity block near exact-name/address labels."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"(\+\d{1,3}\s*\d[\d\s-]{5,}|\(\d{3}\)\s*\d{3}[-\s]?\d{4}|\b\d{3}[-]\d{3}[-]\d{4}\b|\b\d{3}[-]\d{4}[-]\d{4}\b)")
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            text = (span.get("text") or "") + " " + (span.get("text_span") or "")
            if not phone_re.search(text):
                continue
            window = texts[max(0, i-5): min(len(texts), i+6)]
            context = " ".join(((w.get("text") or "") + " " + (w.get("text_span") or "")) for w in window).lower()
            if "exact name of registrant" in context or "address of principal executive offices" in context:
                out.append(span)
        return out
    except Exception:
        return []
