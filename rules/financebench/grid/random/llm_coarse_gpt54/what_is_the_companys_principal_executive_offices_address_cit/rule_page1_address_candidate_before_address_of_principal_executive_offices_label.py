def rule_page1_address_candidate_before_address_of_principal_executive_offices_label(doc: dict) -> list[dict]:
    """Return the span immediately before a standalone address-of-principal-executive-offices label."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(r'\(?Address of principal executive offices\)?', txt, re.I):
                if i > 0:
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
