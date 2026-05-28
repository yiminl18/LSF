def rule_summary_heading_path_suffix(doc: dict) -> list[dict]:
    """Match a summary subject heading whose breadcrumb path ends with the heading text and sits just after SUMMARY."""
    try:
        import re
        out = []
        texts = doc.get("texts", [])
        generic = {"SUMMARY", "COUNSEL", "OPINION", "ORDER", "BACKGROUND", "FILED"}
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            st = span.get("structure") or {}
            path = (st.get("path_text") or "").strip()
            if span.get("page_no") not in {2, 3, 4} or span.get("label") != "section_header" or span.get("bold") != 1 or not txt:
                continue
            if up in generic:
                continue
            if not path or not path.endswith(txt):
                continue
            if "NINTH CIRCUIT" not in path and path != txt:
                continue
            seen = []
            j = i - 1
            while j >= 0 and len(seen) < 3:
                if (texts[j].get("text") or "").strip():
                    seen.append(texts[j])
                j -= 1
            if any(re.search(r"(^|\W)SUMMARY(\W|$)", (s.get("text") or "").upper()) for s in seen):
                out.append(span)
        return out
    except Exception:
        return []
