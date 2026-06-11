def rule_page1_span_before_phone_label(doc: dict) -> list[dict]:
    """Return page-1 spans immediately preceding telephone-number labels in the cover block."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") != 1:
                continue
            if re.search(r"Registrant.?s telephone number", txt, re.I):
                if i > 0:
                    out.append(texts[i - 1])
        return out
    except Exception:
        return []
