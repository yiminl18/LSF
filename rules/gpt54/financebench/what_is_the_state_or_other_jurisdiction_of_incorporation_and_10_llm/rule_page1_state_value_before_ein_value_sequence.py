def rule_page1_state_value_before_ein_value_sequence(doc: dict) -> list[dict]:
    """Match page-1 short value spans occurring before a nearby EIN span in the cover metadata sequence."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            if span.get("page_no") != 1:
                continue
            txt = (span.get("text") or "").strip()
            if 1 <= len(txt.split()) <= 5 and not re.search(r"\d", txt):
                window = texts[i + 1: min(len(texts), i + 6)]
                if any(s.get("page_no") == 1 and re.search(r"\d{2}-\d{7}", (s.get("text") or "").strip()) for s in window):
                    out.append(span)
        return out
    except Exception:
        return []
