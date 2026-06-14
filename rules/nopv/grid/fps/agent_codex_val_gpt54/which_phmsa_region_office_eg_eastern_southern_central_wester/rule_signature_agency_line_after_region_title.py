def rule_signature_agency_line_after_region_title(doc: dict) -> list[dict]:
    """Match the PHMSA agency line that follows a Director/region signature title line."""
    try:
        import re

        title_re = re.compile(
            r"\b(?:Acting\s+)?Director\b[^\n]{0,120}\b(?:Eastern|Southern|Central|Western|Southwest)\b",
            re.I,
        )
        agency_re = re.compile(r"^Pipeline and Hazardous Materials Safety Administration$", re.I)

        texts = doc.get("texts", [])
        out = []
        for i, span in enumerate(texts[:-1]):
            next_span = texts[i + 1]
            text = (span.get("text") or "").strip()
            next_text = (next_span.get("text") or "").strip()
            if span.get("page_no", 0) < 2:
                continue
            if next_span.get("page_no") != span.get("page_no"):
                continue
            if title_re.search(text) and agency_re.match(next_text):
                out.append(next_span)
        return out
    except Exception:
        return []
