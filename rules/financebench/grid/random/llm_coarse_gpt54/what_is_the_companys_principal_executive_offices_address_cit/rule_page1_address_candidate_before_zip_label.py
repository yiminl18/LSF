def rule_page1_address_candidate_before_zip_label(doc: dict) -> list[dict]:
    """Return the span immediately before a '(Zip Code)' label on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r'\(Zip Code\)', txt, re.I):
                if i > 0:
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
