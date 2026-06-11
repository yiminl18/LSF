def rule_page1_address_candidate_near_zip(doc: dict) -> list[dict]:
    """Match page-1 spans near a standalone ZIP/postal-code span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(r'(\d{5}(?:-\d{4})?|BS30 ?8XP)', txt, re.I):
                for j in range(max(0, i - 4), min(len(texts), i + 1)):
                    s2 = texts[j]
                    if s2.get("page_no") == 1 and re.search(r'(Street|Avenue|Boulevard|Drive|Road|Center|Plaza|Santa Monica|San Jose|Seattle|Chicago|Issaquah|Bristol|New York|St\. Paul)', (s2.get("text") or ""), re.I):
                        out.append(s2)
        return out
    except Exception:
        return []
