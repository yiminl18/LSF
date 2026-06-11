def rule_form10q_page1_body_after_quarterly_report(doc: dict) -> list[dict]:
    """Match page-1 body/list spans near quarterly-report language that usually hold the quarter ended line."""
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "quarterly report pursuant to section 13 or 15(d)" in text:
                for j in range(i + 1, min(i + 4, len(texts))):
                    nxt = texts[j]
                    nt = (nxt.get("text") or "").lower()
                    if nxt.get("page_no") == 1 and (
                        "quarterly period ended" in nt or "quarter ended" in nt or "transition period" in nt
                    ):
                        out.append(nxt)
        return out
    except Exception:
        return []
