def rule_page1_body_after_phone_header(doc: dict) -> list[dict]:
    """Match body spans immediately after a phone-number header on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        phone_re = re.compile(r"^\s*(?:\+?\d{1,3}\s*)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}\s*$")
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and phone_re.match((span.get("text", "") or "").strip()):
                for j in range(i, min(len(texts), i + 3)):
                    s = texts[j]
                    if s.get("page_no") == 1:
                        out.append(s)
        return out
    except Exception:
        return []
