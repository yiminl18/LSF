def rule_form10k_page1_body_after_annual_report(doc: dict) -> list[dict]:
    """Match page-1 body/list spans near annual-report language that usually hold the fiscal year ended line."""
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "annual report pursuant to section 13 or 15(d)" in text:
                for j in range(i + 1, min(i + 4, len(texts))):
                    nxt = texts[j]
                    nt = (nxt.get("text") or "").lower()
                    if nxt.get("page_no") == 1 and (
                        "fiscal year ended" in nt or "transition period" in nt
                    ):
                        out.append(nxt)
        return out
    except Exception:
        return []
