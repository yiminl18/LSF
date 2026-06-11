def rule_page1_exchange_after_name_of_exchange_label(doc: dict) -> list[dict]:
    """Return the first few non-label spans after a 'Name of each exchange...' label on page 1."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            low = (span.get("text") or "").lower()
            if "name of each exchange on which registered" in low or "name of each exchange on which registered" in (span.get("text_span") or "").lower():
                for j in range(i + 1, min(i + 5, len(texts))):
                    s = texts[j]
                    if s.get("page_no") != 1:
                        break
                    stxt = (s.get("text") or "").strip().lower()
                    if stxt and "trading symbol" not in stxt and "title of each class" not in stxt:
                        out.append(s)
        return out
    except Exception:
        return []
