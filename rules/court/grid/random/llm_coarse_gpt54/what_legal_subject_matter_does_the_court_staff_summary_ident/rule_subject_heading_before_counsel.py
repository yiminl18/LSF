def rule_subject_heading_before_counsel(doc: dict) -> list[dict]:
    """Match the last non-COUNSEL heading before the COUNSEL section near the top, often the summary subject."""
    try:
        texts = doc.get("texts", [])
        counsel_idx = None
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().upper()
            if "COUNSEL" == txt or txt.startswith("COUNSEL"):
                counsel_idx = i
                break
        if counsel_idx is None:
            return []
        candidates = []
        for i in range(max(0, counsel_idx - 12), counsel_idx):
            s = texts[i]
            txt = (s.get("text") or "").strip()
            up = txt.upper()
            if s.get("label") == "section_header" and txt:
                if "SUMMARY" in up or "OPINION" in up or "BACKGROUND" in up or "COUNSEL" in up:
                    continue
                candidates.append(s)
        return candidates[-1:] if candidates else []
    except Exception:
        return []
