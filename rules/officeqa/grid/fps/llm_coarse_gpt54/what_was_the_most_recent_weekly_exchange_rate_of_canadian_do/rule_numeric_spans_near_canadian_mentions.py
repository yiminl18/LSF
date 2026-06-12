def rule_numeric_spans_near_canadian_mentions(doc: dict) -> list[dict]:
    """Match spans with decimal numbers near Canadian dollar position mentions."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if "canadian dollar positions" in txt or "fcp-ii-" in txt:
                for j in range(max(0, i - 3), min(len(texts), i + 6)):
                    st = texts[j]
                    stxt = st.get("text") or ""
                    if re.search(r"\b\d+\.\d{3,4}\b", stxt):
                        out.append(st)
        return out
    except Exception:
        return []
