def rule_page12_text_phone_next_cue(doc: dict) -> list[dict]:
    """Match page-1/2 text spans with a phone value when the next span carries the cue."""
    try:
        import re

        phone_re = re.compile(r"\+?\d[\d\s().-]{7,}\d|\(\d{3}\)\s*\d{3}[- ]?\d{4}|\d{3}[- ]\d{3}[- ]\d{4}|\d{10,15}")
        cue_terms = ("telephone number", "including area code", "registrant's telephone", "registrant’s telephone")

        texts = list(enumerate(doc.get("texts", [])))
        texts = [(i, s) for i, s in texts if (s.get("page_no") or 999) <= 2]
        texts.sort(key=lambda x: ((x[1].get("page_no") or 0), x[0]))

        out = []
        seen = set()
        for pos, (_, span) in enumerate(texts):
            if span.get("label") != "text":
                continue
            text = (span.get("text") or "").replace("\n", " ").strip()
            match = phone_re.search(text)
            if not match or sum(ch.isdigit() for ch in match.group(0)) < 10:
                continue

            page = span.get("page_no")
            cue_hits = []
            for off in (1, 2):
                if pos + off >= len(texts):
                    continue
                _, other = texts[pos + off]
                if other.get("page_no") != page:
                    continue
                other_low = ((other.get("text") or "").replace("\n", " ").strip()).lower()
                if any(term in other_low for term in cue_terms):
                    cue_hits.append(other)

            if not cue_hits:
                continue

            for item in [span] + cue_hits:
                key = id(item)
                if key not in seen:
                    seen.add(key)
                    out.append(item)
        return out
    except Exception:
        return []
