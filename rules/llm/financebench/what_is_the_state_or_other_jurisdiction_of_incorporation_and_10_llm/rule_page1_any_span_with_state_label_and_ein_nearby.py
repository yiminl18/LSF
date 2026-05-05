def rule_page1_any_span_with_state_label_and_ein_nearby(doc: dict) -> list[dict]:
    """Match page-1 spans with state/incorporation labels when an EIN appears within a nearby window."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"state|jurisdiction|incorporation", txt, re.I):
                window = texts[max(0, i - 4): min(len(texts), i + 5)]
                if any(s.get("page_no") == 1 and re.search(r"\d{2}-\d{7}", ((s.get("text") or "") + " " + (s.get("text_span") or ""))) for s in window):
                    out.append(span)
        return out
    except Exception:
        return []
