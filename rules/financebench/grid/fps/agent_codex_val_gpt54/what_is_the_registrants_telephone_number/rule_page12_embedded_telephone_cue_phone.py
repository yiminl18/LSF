def rule_page12_embedded_telephone_cue_phone(doc: dict) -> list[dict]:
    """Match page-1/2 spans where the telephone cue and phone value appear together."""
    try:
        import re

        cue_terms = (
            "telephone number",
            "including area code",
            "registrant's telephone",
            "registrant’s telephone",
        )
        phone_re = re.compile(r"\+?\d[\d\s().-]{7,}\d|\(\d{3}\)\s*\d{3}[- ]?\d{4}|\d{3}[- ]\d{3}[- ]\d{4}")

        out = []
        seen = set()
        for span in doc.get("texts", []):
            if (span.get("page_no") or 999) > 2:
                continue
            if span.get("label") not in {"text", "section_header"}:
                continue
            text = (span.get("text") or "").replace("\n", " ").strip()
            low = text.lower()
            if not any(term in low for term in cue_terms):
                continue
            match = phone_re.search(text)
            if not match:
                continue
            if sum(ch.isdigit() for ch in match.group(0)) < 10:
                continue
            key = id(span)
            if key not in seen:
                seen.add(key)
                out.append(span)
        return out
    except Exception:
        return []
