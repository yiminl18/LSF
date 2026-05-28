def rule_summary_subject_h1_after_summary_text(doc: dict) -> list[dict]:
    """Match H1 section headers that appear immediately after a standalone SUMMARY text span."""
    try:
        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip().upper()
            if txt == "SUMMARY" or txt.endswith("SUMMARY") or "SUMMARY" in txt:
                for j in range(i + 1, min(i + 6, len(texts))):
                    s = texts[j]
                    st = s.get("structure") or {}
                    stxt = (s.get("text") or "").strip()
                    if st.get("level") == "H1" and stxt and stxt.upper() not in {"COUNSEL", "OPINION", "BACKGROUND"}:
                        out.append(s)
                        break
        return out
    except Exception:
        return []
