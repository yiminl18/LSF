def rule_summary_heading_short_h1_h2(doc: dict) -> list[dict]:
    """Match a short H1 or H2 summary subject section header near SUMMARY on pages 2-3."""
    try:
        import re
        out = []
        texts = doc.get("texts", [])
        generic = {"SUMMARY", "COUNSEL", "OPINION", "ORDER", "BACKGROUND", "FILED"}
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            st = span.get("structure") or {}
            if span.get("page_no") not in {2, 3} or span.get("label") != "section_header" or not txt:
                continue
            if up in generic:
                continue
            if (st.get("level") or "") not in {"H1", "H2"} or len(txt.split()) > 8:
                continue
            seen = []
            j = i - 1
            while j >= 0 and len(seen) < 2:
                if (texts[j].get("text") or "").strip():
                    seen.append(texts[j])
                j -= 1
            if any(re.search(r"(^|\W)SUMMARY(\W|$)", (s.get("text") or "").upper()) for s in seen):
                out.append(span)
        return out
    except Exception:
        return []
