def rule_summary_subject_h2_under_court_header(doc: dict) -> list[dict]:
    """Match H2 section headers near the top under the court header that are likely summary subject labels."""
    try:
        out = []
        for span in doc.get("texts", []):
            txt = (span.get("text") or "").strip()
            st = span.get("structure") or {}
            level = st.get("level")
            path = (st.get("path_text") or "").upper()
            if level == "H2" and txt and "NINTH CIRCUIT" in path:
                up = txt.upper()
                if "COUNSEL" in up or "OPINION" in up or "BACKGROUND" in up or "SUMMARY" in up:
                    continue
                if span.get("page_no") in {1, 2, 3}:
                    out.append(span)
        return out
    except Exception:
        return []
