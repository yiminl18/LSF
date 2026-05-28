def rule_summary_subject_bold_after_summary(doc: dict) -> list[dict]:
    """Match the first bold non-SUMMARY span after a SUMMARY marker on page 2 or nearby pages."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if "SUMMARY" in txt.upper():
                base_page = span.get("page_no")
                for j in range(i + 1, min(i + 10, len(texts))):
                    s = texts[j]
                    stxt = (s.get("text") or "").strip()
                    if not stxt:
                        continue
                    if s.get("page_no") not in {base_page, (base_page or 0) + 1}:
                        continue
                    if s.get("bold") == 1 and "SUMMARY" not in stxt.upper() and "COUNSEL" not in stxt.upper() and "OPINION" not in stxt.upper():
                        out.append(s)
                        break
        return out
    except Exception:
        return []
