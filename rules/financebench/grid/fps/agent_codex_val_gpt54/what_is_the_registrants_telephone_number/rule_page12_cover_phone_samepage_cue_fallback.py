def rule_page12_cover_phone_samepage_cue_fallback(doc: dict) -> list[dict]:
    """Match page-1/2 cover-page phone spans when the telephone cue exists elsewhere on that page."""
    try:
        import re

        phone_re = re.compile(r"\+?\d[\d\s().-]{7,}\d|\(\d{3}\)\s*\d{3}[- ]?\d{4}|\d{3}[- ]\d{3}[- ]\d{4}|\d{10,15}")
        cue_terms = ("telephone", "telephone number", "including area code", "registrant's telephone", "registrant’s telephone")
        bad_terms = ("item ", "part ", "news release", "investor relations", "media contacts", "conference call")

        by_page = {}
        for span in doc.get("texts", []):
            page = span.get("page_no") or 999
            if page > 2:
                continue
            by_page.setdefault(page, []).append(span)

        out = []
        seen = set()
        for page, spans in by_page.items():
            if not any(any(term in ((s.get("text") or "").lower()) for term in cue_terms) for s in spans):
                continue
            for span in spans:
                if span.get("label") not in {"text", "section_header"}:
                    continue
                text = (span.get("text") or "").replace("\n", " ").strip()
                low = text.lower()
                path_low = ((span.get("structure") or {}).get("path_text") or "").lower()
                if any(term in low for term in cue_terms):
                    continue
                if any(term in path_low for term in bad_terms):
                    continue
                match = phone_re.fullmatch(text) or phone_re.search(text)
                if not match or sum(ch.isdigit() for ch in match.group(0)) < 10:
                    continue
                if text.isdigit():
                    continue
                if span.get("label") == "section_header" or span.get("bold") == 1 or len(text) <= 40:
                    key = id(span)
                    if key not in seen:
                        seen.add(key)
                        out.append(span)
        return out
    except Exception:
        return []
