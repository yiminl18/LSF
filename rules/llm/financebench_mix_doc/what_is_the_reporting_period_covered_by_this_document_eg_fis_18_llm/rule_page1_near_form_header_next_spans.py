def rule_page1_near_form_header_next_spans(doc: dict) -> list[dict]:
    """Return spans immediately following a FORM 10-K/10-Q/8-K header if they contain period cues."""
    import re
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().upper()
            if span.get("page_no") == 1 and txt in {"FORM 10-K", "FORM 10-Q", "FORM 8-K"}:
                for j in range(i + 1, min(i + 8, len(texts))):
                    s = texts[j]
                    t = ((s.get("text") or "") + " " + (s.get("text_span") or "")).lower()
                    if re.search(r'(fiscal year ended|quarterly period ended|date of report|transition period)', t):
                        out.append(s)
        return out
    except Exception:
        return []
