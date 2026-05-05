def rule_page1_registration_sentences_with_registered(doc: dict) -> list[dict]:
    """Match page 1 spans containing 'registered' plus exchange/symbol cues."""
    try:
        import re
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "")
            if span.get("page_no") == 1 and re.search(r"registered", txt, re.I) and re.search(r"symbol|exchange|nasdaq|new york stock exchange|nyse", txt, re.I):
                out.append(span)
        return out
    except Exception:
        return []
