def rule_subject_heading_after_summary_before_counsel(doc: dict) -> list[dict]:
    """Match headings between SUMMARY and COUNSEL, excluding generic labels, as likely subject matter headings."""
    try:
        texts = doc.get("texts", [])
        summary_idx = None
        counsel_idx = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().upper()
            if summary_idx is None and "SUMMARY" in txt:
                summary_idx = i
            if counsel_idx is None and "COUNSEL" in txt:
                counsel_idx = i
        if summary_idx is None:
            return []
        end = counsel_idx if counsel_idx is not None else min(len(texts), summary_idx + 15)
        out = []
        for i in range(summary_idx + 1, end):
            s = texts[i]
            txt = (s.get("text") or "").strip()
            up = txt.upper()
            if s.get("label") == "section_header" and txt:
                if "OPINION" in up or "BACKGROUND" in up or "COUNSEL" in up:
                    continue
                out.append(s)
        return out
    except Exception:
        return []
