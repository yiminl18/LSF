def rule_page1_ein_value_after_state_value_sequence(doc: dict) -> list[dict]:
    """Match page-1 EIN spans that occur shortly after a short non-numeric state/jurisdiction value span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.fullmatch(r"\d{2}-\d{7}", txt):
                prevs = texts[max(0, i - 5): i]
                if any(
                    s.get("page_no") == 1 and 1 <= len((s.get("text") or "").strip().split()) <= 5 and not re.search(r"\d", (s.get("text") or "").strip())
                    for s in prevs
                ):
                    out.append(span)
        return out
    except Exception:
        return []
