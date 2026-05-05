def rule_before_securities_registered(doc: dict) -> list[dict]:
    """Match spans immediately before the securities-registered section on page 1."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"securities registered pursuant to section 12\(b\)", text, re.I):
                for j in range(max(0, i - 3), i + 1):
                    cand = texts[j]
                    if cand.get("page_no") == 1:
                        out.append(cand)
        seen = set()
        dedup = []
        for s in out:
            key = id(s)
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        return dedup
    except Exception:
        return []
