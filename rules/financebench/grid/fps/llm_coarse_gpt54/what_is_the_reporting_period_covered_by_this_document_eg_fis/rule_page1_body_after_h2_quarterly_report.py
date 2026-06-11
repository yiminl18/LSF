def rule_page1_body_after_h2_quarterly_report(doc: dict) -> list[dict]:
    """Match body spans under quarterly-report paths that contain the quarter ended line."""
    try:
        out = []
        for span in doc.get("texts", []):
            path = ((span.get("structure") or {}).get("path_text") or "").lower()
            if span.get("page_no") == 1 and "quarterly report pursuant to section 13 or 15(d)" in path:
                text = (span.get("text") or "").lower()
                if "quarterly period ended" in text or "quarter ended" in text:
                    out.append(span)
        return out
    except Exception:
        return []
