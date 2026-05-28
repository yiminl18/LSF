def rule_summary_heading_followed_by_body_text(doc: dict) -> list[dict]:
    """Match a summary subject heading that is followed by ordinary body text and preceded nearby by SUMMARY."""
    try:
        import re
        out = []
        texts = doc.get("texts", [])
        generic = {"SUMMARY", "COUNSEL", "OPINION", "ORDER", "BACKGROUND", "FILED"}
        for i, span in enumerate(texts):
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            if span.get("page_no") not in {2, 3} or span.get("label") != "section_header" or span.get("bold") != 1 or not txt:
                continue
            if up in generic:
                continue
            nxt = texts[i + 1] if i + 1 < len(texts) else None
            if not nxt or nxt.get("label") != "text" or nxt.get("bold") == 1:
                continue
            if nxt.get("page_no") not in {span.get("page_no"), (span.get("page_no") or 0) + 1}:
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
