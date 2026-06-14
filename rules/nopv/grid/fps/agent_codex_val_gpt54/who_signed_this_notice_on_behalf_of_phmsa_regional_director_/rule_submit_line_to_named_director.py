def rule_submit_line_to_named_director(doc: dict) -> list[dict]:
    """Match short compliance-order lines that direct submissions to the named PHMSA director."""
    try:
        import re

        submit_re = re.compile(
            r"\bsubmit\b[^\n]{0,220}\bto\b[^\n]{0,80}\b(?:Acting\s+)?Director\b[^\n]{0,120}\b(?:Eastern|Southern|Central|Western|Southwest)\b",
            re.I,
        )

        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no", 0) < 3:
                continue
            if span.get("label") not in ("text", "list_item"):
                continue
            if submit_re.search(text) and len(text) <= 420:
                out.append(span)
        return out
    except Exception:
        return []
