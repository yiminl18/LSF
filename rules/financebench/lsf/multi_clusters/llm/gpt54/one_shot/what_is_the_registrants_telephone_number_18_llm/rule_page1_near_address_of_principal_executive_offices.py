def rule_page1_near_address_of_principal_executive_offices(doc: dict) -> list[dict]:
    """Match page-1 spans near the principal executive offices label, where the phone often appears."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = span.get("text", "") or ""
            if span.get("page_no") == 1 and re.search(r"address of principal executive offices", text, re.I):
                for j in range(max(0, i - 3), min(len(texts), i + 4)):
                    s = texts[j]
                    blob = (s.get("text", "") or "") + " " + (s.get("text_span", "") or "")
                    if s.get("page_no") == 1 and re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}", blob):
                        out.append(s)
        return out
    except Exception:
        return []
