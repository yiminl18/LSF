def rule_page1_incorporation_label_followed_by_ein_value(doc: dict) -> list[dict]:
    """Match page-1 spans immediately after an incorporation label when they look like EIN values or combined answer lines."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            cur_text = (span.get("text") or "").strip()
            nxt = texts[i + 1]
            nxt_text = (nxt.get("text") or "").strip()
            if span.get("page_no") == 1 and nxt.get("page_no") == 1:
                if re.search(r"state( or other jurisdiction)? of incorporation( or organization)?|state of incorporation", cur_text, re.I):
                    if re.search(r"\d{2}-\d{7}", nxt_text) or len(nxt_text.split()) <= 6:
                        out.append(nxt)
        return out
    except Exception:
        return []
