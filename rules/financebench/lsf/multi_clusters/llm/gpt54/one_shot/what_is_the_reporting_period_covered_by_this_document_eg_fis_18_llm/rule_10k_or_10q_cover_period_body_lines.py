def rule_10k_or_10q_cover_period_body_lines(doc: dict) -> list[dict]:
    """Match body lines on the cover page that are children of FORM 10-K/10-Q and contain the answer."""
    import re
    try:
        out = []
        for span in doc.get("texts", []):
            path = (span.get("structure", {}) or {}).get("path_text", "")
            lvl = (span.get("structure", {}) or {}).get("level", "")
            txt = (span.get("text") or "").lower()
            if span.get("page_no") == 1 and lvl == "Body" and ("FORM 10-K" in path or "FORM 10-Q" in path):
                if re.search(r'for the (fiscal year|quarterly period) ended', txt):
                    out.append(span)
        return out
    except Exception:
        return []
