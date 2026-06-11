def rule_page1_commission_file_preceding_reporting_line(doc: dict) -> list[dict]:
    """Match page-1 spans immediately preceding a Commission File Number line when they contain reporting-period language."""
    try:
        out = []
        texts = doc.get("texts", [])
        for i, span in enumerate(texts):
            text = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and "commission file" in text:
                for j in [i - 1, i - 2]:
                    if 0 <= j < len(texts):
                        cand = texts[j]
                        ct = (cand.get("text") or "").lower()
                        if cand.get("page_no") == 1 and any(k in ct for k in [
                            "fiscal year ended", "quarterly period ended", "date of report", "transition period"
                        ]):
                            out.append(cand)
        return out
    except Exception:
        return []
