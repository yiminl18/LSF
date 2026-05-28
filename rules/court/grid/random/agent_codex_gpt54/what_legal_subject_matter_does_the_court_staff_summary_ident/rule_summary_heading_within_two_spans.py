def rule_summary_heading_within_two_spans(doc: dict) -> list[dict]:
    """Match a summary subject section header whose two nearest prior spans include a standalone SUMMARY marker."""
    try:
        import re
        out = []
        texts = doc.get("texts", [])
        generic = {"SUMMARY", "COUNSEL", "OPINION", "ORDER", "BACKGROUND", "FILED"}
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            st = span.get("structure") or {}
            if span.get("page_no") not in {2, 3} or span.get("label") != "section_header" or span.get("bold") != 1 or not txt:
                continue
            if up in generic:
                continue
            seen = []
            j = i - 1
            while j >= 0 and len(seen) < 2:
                if (texts[j].get("text") or "").strip():
                    seen.append(texts[j])
                j -= 1
            if not any(re.search(r"(^|\W)SUMMARY(\W|$)", (s.get("text") or "").upper()) for s in seen):
                continue
            path = (st.get("path_text") or "").strip()
            if not path or path.split(" | ")[-1].strip() == txt:
                out.append(span)
        return out
    except Exception:
        return []
