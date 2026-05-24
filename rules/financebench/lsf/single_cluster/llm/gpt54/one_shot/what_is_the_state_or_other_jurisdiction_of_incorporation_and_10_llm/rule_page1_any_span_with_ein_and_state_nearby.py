def rule_page1_any_span_with_ein_and_state_nearby(doc: dict) -> list[dict]:
    """Match page-1 spans with EIN numbers when a state/incorporation label appears within a nearby window."""
    try:
        import re
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = ((span.get("text") or "") + " " + (span.get("text_span") or "")).strip()
            if span.get("page_no") == 1 and re.search(r"\d{2}-\d{7}", txt):
                window = texts[max(0, i - 4): min(len(texts), i + 5)]
                if any(s.get("page_no") == 1 and re.search(r"state|jurisdiction|incorporation", ((s.get("text") or "") + " " + (s.get("text_span") or "")), re.I) for s in window):
                    out.append(span)
        return out
    except Exception:
        return []
