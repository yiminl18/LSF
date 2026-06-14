def rule_signature_title_line_with_region(doc: dict) -> list[dict]:
    """Match short later-page signature lines naming the Director and PHMSA region."""
    try:
        import re

        region_re = re.compile(
            r"\b(?:Eastern|Southern|Central|Western|Southwest)\b",
            re.I,
        )
        title_re = re.compile(r"\b(?:Acting\s+)?Director\b", re.I)
        agency_re = re.compile(
            r"\b(?:Office of Pipeline Safety|OPS|Pipeline and Hazardous Materials Safety Administration)\b",
            re.I,
        )

        out = []
        for span in doc.get("texts", []):
            text = (span.get("text") or "").strip()
            if span.get("page_no", 0) < 2:
                continue
            if span.get("label") not in ("text", "table"):
                continue
            if not title_re.search(text) or not region_re.search(text) or not agency_re.search(text):
                continue
            if span.get("label") == "table" or len(text) <= 220:
                out.append(span)
        return out
    except Exception:
        return []
