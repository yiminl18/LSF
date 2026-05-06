def rule_page1_before_state_header(doc: dict) -> list[dict]:
    """Match the nearest preceding page-1 span before a state-of-incorporation header/caption."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") != 1:
                continue
            if "state or other jurisdiction of incorporation" in txt or "state of incorporation" in txt:
                for j in range(i - 1, max(-1, i - 4), -1):
                    if j >= 0 and texts[j].get("page_no") == 1:
                        cand = texts[j]
                        if any(ch.isalpha() for ch in (cand.get("text") or "")):
                            out.append(cand)
                            break
        return out
    except Exception:
        return []
