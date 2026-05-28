def rule_subject_heading_under_summary_page2_first(doc: dict) -> list[dict]:
    """Match the first bold heading on page 2 after SUMMARY, the most common location for the answer."""
    try:
        texts = doc.get("texts", [])
        out = []
        seen_summary = False
        for span in texts:
            txt = (span.get("text") or "").strip()
            up = txt.upper()
            if span.get("page_no") == 2 and "SUMMARY" in up:
                seen_summary = True
                continue
            if seen_summary and span.get("page_no") == 2:
                if span.get("bold") == 1 and txt and all(bad not in up for bad in ["COUNSEL", "OPINION", "BACKGROUND"]):
                    out.append(span)
                    break
        return out
    except Exception:
        return []
