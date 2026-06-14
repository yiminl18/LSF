def rule_submit_line_to_region_director(doc: dict) -> list[dict]:
    """Match short compliance-order lines that direct submissions to the named PHMSA region director."""
    try:
        import re

        submit_re = re.compile(
            r"\b(?:submit|provide|requested?)\b[^\n]{0,220}\bDirector\b[^\n]{0,80}\b(?:Eastern|Southern|Central|Western|Southwest)\b",
            re.I,
        )

        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no", 0) < 3:
                continue
            if span.get("label") not in ("text", "list_item"):
                continue
            if submit_re.search(text) and len(text) <= 320:
                out.append(span)
        return out
    except Exception:
        return []
