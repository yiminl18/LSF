def rule_summary_subject_section_header(doc: dict) -> list[dict]:
    """Match bold section_header spans immediately after a SUMMARY marker, which usually contain the legal subject."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            if "SUMMARY" in txt.upper():
                for j in range(i + 1, min(i + 8, len(texts))):
                    s = texts[j]
                    stxt = (s.get("text") or "").strip()
                    if s.get("label") == "section_header" and stxt and "COUNSEL" not in stxt.upper() and "OPINION" not in stxt.upper():
                        out.append(s)
                        break
        return out
    except Exception:
        return []
