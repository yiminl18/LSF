def rule_page1_form_header_or_neighbor(doc: dict) -> list[dict]:
    """Match FORM header spans and their immediate neighbors when any contain reporting-period language."""
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and text.startswith("form "):
                window = texts[max(0, i - 1): min(len(texts), i + 4)]
                if any(any(k in (w.get("text") or "").lower() for k in [
                    "fiscal year ended", "quarterly period ended", "date of report", "for the period ending"
                ]) for w in window):
                    out.extend(window)
        return out
    except Exception:
        return []
