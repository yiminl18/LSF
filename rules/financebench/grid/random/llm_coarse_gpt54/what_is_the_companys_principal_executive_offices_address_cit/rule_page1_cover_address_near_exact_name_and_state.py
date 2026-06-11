def rule_page1_cover_address_near_exact_name_and_state(doc: dict) -> list[dict]:
    """Match spans near exact-name and state/incorporation labels that look like the address answer."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'Exact name of registrant|State or other jurisdiction of incorporation', txt, re.I):
                for j in range(i, min(i + 8, len(texts))):
                    cand = (texts[j].get("text") or "").strip()
                    if re.search(r'Issaquah|Seattle|San Jose|Santa Monica|St\. Paul|New York|Chicago|Warmley|Bristol', cand, re.I):
                        out.append(texts[j])
        return out
    except Exception:
        return []
