def rule_form8k_page1_body_after_current_report(doc: dict) -> list[dict]:
    """Match page-1 spans following CURRENT REPORT / Securities Exchange Act language that usually hold the event date."""
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and (
                "current report" in text or
                "pursuant to section 13 or 15(d) of the securities exchange act of 1934" in text
            ):
                for j in range(i + 1, min(i + 5, len(texts))):
                    nxt = texts[j]
                    nt = (nxt.get("text") or "").lower()
                    if nxt.get("page_no") == 1 and "date of report" in nt:
                        out.append(nxt)
        return out
    except Exception:
        return []
