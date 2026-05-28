def rule_summary_subject_first_child_after_summary(doc: dict) -> list[dict]:
    """Match the first child heading-like span after SUMMARY, which often is the subject matter label."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().upper()
            if "SUMMARY" in txt:
                for j in range(i + 1, min(i + 8, len(texts))):
                    s = texts[j]
                    stxt = (s.get("text") or "").strip()
                    if not stxt:
                        continue
                    if s.get("label") in {"section_header", "text"} and s.get("bold") == 1:
                        up = stxt.upper()
                        if "COUNSEL" in up or "OPINION" in up or "BACKGROUND" in up:
                            continue
                        out.append(s)
                        break
        return out
    except Exception:
        return []
