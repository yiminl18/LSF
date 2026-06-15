def rule_page12_phone_previous_cue_window(doc: dict) -> list[dict]:
    """Match page-1/2 phone spans whose preceding cover-page window carries the cue."""
    try:
        import re

        phone_re = re.compile(r"\+?\d[\d\s().-]{7,}\d|\(\d{3}\)\s*\d{3}[- ]?\d{4}|\d{3}[- ]\d{3}[- ]\d{4}|\d{10,15}")
        cue_terms = ("telephone", "telephone number", "including area code", "registrant's telephone", "registrant’s telephone")

        texts = list(enumerate(doc.get("texts", [])))
        texts = [(i, s) for i, s in texts if (s.get("page_no") or 999) <= 2]
        texts.sort(key=lambda x: ((x[1].get("page_no") or 0), x[0]))

        out = []
        seen = set()
        for pos, (_, span) in enumerate(texts):
            if span.get("label") not in {"text", "section_header"}:
                continue
            text = (span.get("text") or "").replace("\n", " ").strip()
            match = phone_re.fullmatch(text) or phone_re.search(text)
            if not match or sum(ch.isdigit() for ch in match.group(0)) < 10:
                continue
            digits_only = text.isdigit()

            page = span.get("page_no")
            cue_hits = []
            closest_off = None
            for off in (1, 2, 3):
                if pos - off < 0:
                    continue
                _, other = texts[pos - off]
                if other.get("page_no") != page:
                    continue
                other_low = ((other.get("text") or "").replace("\n", " ").strip()).lower()
                if any(term in other_low for term in cue_terms):
                    cue_hits.append(other)
                    if closest_off is None or off < closest_off:
                        closest_off = off

            if not cue_hits:
                continue
            if digits_only and closest_off not in (1,):
                continue

            for item in cue_hits + [span]:
                key = id(item)
                if key not in seen:
                    seen.add(key)
                    out.append(item)
        return out
    except Exception:
        return []
