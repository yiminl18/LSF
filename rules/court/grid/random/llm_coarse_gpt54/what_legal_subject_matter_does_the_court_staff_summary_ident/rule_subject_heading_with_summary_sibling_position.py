def rule_subject_heading_with_summary_sibling_position(doc: dict) -> list[dict]:
    """Match heading spans with low sibling_index_norm after SUMMARY, indicating early summary subject placement."""
    try:
        out = []
        for span in doc.get("texts", []):
            st = span.get("structure") or {}
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            sib = st.get("sibling_index_norm")
            if txt and span.get("bold") == 1 and span.get("page_no") in {1, 2, 3}:
                if any(bad in up for bad in ["SUMMARY", "COUNSEL", "OPINION", "BACKGROUND"]):
                    continue
                if isinstance(sib, (int, float)) and sib <= 0.8:
                    out.append(span)
        return out
    except Exception:
        return []
