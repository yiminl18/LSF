def rule_page1_activision_santa_monica_split(doc: dict) -> list[dict]:
    """Match Activision-style split page-1 address components for Santa Monica, CA."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if re.search(r'Santa Monica,', txt, re.I):
                out.append(span)
                for j in range(i + 1, min(i + 3, len(texts))):
                    if re.fullmatch(r'CA', (texts[j].get("text") or "").strip(), re.I):
                        out.append(texts[j])
        return out
    except Exception:
        return []
