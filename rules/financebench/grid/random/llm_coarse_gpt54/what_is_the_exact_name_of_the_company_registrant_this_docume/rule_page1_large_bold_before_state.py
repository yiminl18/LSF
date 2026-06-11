def rule_page1_large_bold_before_state(doc: dict) -> list[dict]:
    """Match the nearest previous large bold span before a page-1 state-of-incorporation label."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "state or other jurisdiction" in txt:
                for j in range(i - 1, -1, -1):
                    prev = texts[j]
                    if (
                        prev.get("page_no") == 1
                        and prev.get("bold") == 1
                        and float(prev.get("size") or 0) >= 9
                    ):
                        out.append(prev)
                        break
        return out
    except Exception:
        return []
