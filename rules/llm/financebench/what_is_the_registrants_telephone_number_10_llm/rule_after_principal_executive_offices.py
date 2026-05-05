def rule_after_principal_executive_offices(doc: dict) -> list[dict]:
    """Match spans near the principal executive offices label, where the telephone number usually appears."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            text = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"principal executive offices", text, re.I):
                for j in range(max(0, i - 1), min(len(texts), i + 4)):
                    cand = texts[j]
                    if cand.get("page_no") == 1:
                        out.append(cand)
        # dedupe by object identity fallback via id
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
