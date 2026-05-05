def rule_page1_window_after_incorporation_label(doc: dict) -> list[dict]:
    """Match a small page-1 window after the incorporation label to capture nearby state and EIN spans."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if span.get("page_no") == 1 and re.search(r"state( or other jurisdiction)? of incorporation( or organization)?|state of incorporation", txt, re.I):
                for s in texts[max(0, i - 2): min(len(texts), i + 5)]:
                    if s.get("page_no") == 1:
                        out.append(s)
        return out
    except Exception:
        return []
