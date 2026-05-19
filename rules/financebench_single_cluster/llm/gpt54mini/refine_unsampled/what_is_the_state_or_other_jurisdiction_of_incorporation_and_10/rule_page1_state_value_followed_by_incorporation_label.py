def rule_page1_state_value_followed_by_incorporation_label(doc: dict) -> list[dict]:
    """Match page-1 spans immediately preceding an incorporation-label span."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            nxt = texts[i + 1]
            nxt_text = (nxt.get("text") or "").strip()
            cur_text = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and nxt.get("page_no") == 1:
                if re.search(r"state( or other jurisdiction)? of incorporation( or organization)?|state of incorporation", nxt_text, re.I):
                    if cur_text and not re.search(r"\d{2}-\d{7}", cur_text):
                        out.append(span)
        return out
    except Exception:
        return []
