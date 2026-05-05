def rule_page1_near_exact_name_of_registrant(doc: dict) -> list[dict]:
    """Match page-1 spans near the exact-name-of-registrant label, within the cover-page identity block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") == 1 and re.search(r"exact name of registrant", span.get("text", "") or "", re.I):
                for j in range(i, min(len(texts), i + 12)):
                    s = texts[j]
                    blob = (s.get("text", "") or "") + " " + (s.get("text_span", "") or "")
                    if s.get("page_no") == 1 and (
                        re.search(r"telephone number|area code", blob, re.I)
                        or re.search(r"^\s*(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s\-)]*\d{3,4}[\s\-]?\d{4,}\s*$", (s.get("text", "") or "").strip())
                    ):
                        out.append(s)
        return out
    except Exception:
        return []
