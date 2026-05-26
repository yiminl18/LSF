import re


def rule_registrant_exact_name(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        paragraphs = doc.get("paragraphs") or []

        ordered = sorted(
            [x for x in lines if isinstance(x, dict)],
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        def clean(text: str) -> str:
            return " ".join((text or "").replace("\u00a0", " ").split())

        def make_span(source: dict, text: str) -> list[dict]:
            span = {"text": text}
            if source.get("page_no") is not None:
                span["page_no"] = source.get("page_no")
            if source.get("line_no") is not None:
                span["line_no"] = source.get("line_no")
            if source.get("paragraph_no") is not None:
                span["paragraph_no"] = source.get("paragraph_no")
            return [span]

        label_pat = re.compile(
            r"(?i)\b(?:exact name of registrant|name of registrant as specified in its charter)\b"
        )
        signature_pat = re.compile(r"(?i)\(registrant\)\s*$")
        boilerplate_pat = re.compile(
            r"(?i)\b(?:"
            r"table of contents|"
            r"commission file number|commission file no\.?|"
            r"form\s+10-k|form\s+10-q|form\s+8-k|"
            r"annual report|quarterly report|current report|"
            r"pursuant to section\s+13|"
            r"pursuant to section\s+15\(d\)|"
            r"securities exchange act of 1934|"
            r"indicate by check mark|"
            r"registrant'?s telephone number|"
            r"address of principal executive offices|"
            r"state or other jurisdiction|"
            r"former name or former address|"
            r"if changed since last report|"
            r"item\s+\d+\.|"
            r"documents incorporated by reference|"
            r"pursuant to the requirements of the securities exchange act"
            r")\b"
        )

        def strip_label_prefix(text: str) -> str:
            t = clean(text)
            if not t:
                return ""
            m = re.search(
                r"(?i)^(.*?)(?:\s*\(?\s*(?:exact name of registrant|name of registrant as specified in its charter)\b.*$)",
                t,
            )
            if m:
                prefix = clean(m.group(1))
                if prefix:
                    return prefix
            if t.endswith("(Registrant)"):
                prefix = clean(t[: t.lower().rfind("(registrant)")])
                if prefix:
                    return prefix
            return t

        def looks_like_name(text: str) -> bool:
            t = clean(text)
            if not t or len(t) > 120:
                return False
            if boilerplate_pat.search(t):
                return False
            if not re.search(r"[A-Za-z]", t):
                return False
            if re.search(r"\b(?:state|delaware|california|nevada|washington|texas|new york|florida|colorado|michigan|illinois|virginia|massachusetts|ohio|utah)\b", t, re.I):
                return False
            if re.search(r"\b(?:zip|address|telephone|incorporation|organization|employer identification|identification no|principal executive)\b", t, re.I):
                return False
            if t.count(",") > 3:
                return False
            digit_count = sum(ch.isdigit() for ch in t)
            if digit_count > 3 and not re.search(r"\b(?:3m|1st|2nd|4th)\b", t, re.I):
                return False
            return True

        def find_previous_candidate(start_idx: int) -> tuple[str, dict] | None:
            for j in range(start_idx - 1, max(-1, start_idx - 6), -1):
                source = ordered[j]
                text = clean(source.get("text"))
                if not text:
                    continue
                if boilerplate_pat.search(text):
                    continue
                candidate = strip_label_prefix(text)
                if candidate and looks_like_name(candidate):
                    return candidate, source
            return None

        # First pass: exact-name label on the cover page.
        for i, item in enumerate(ordered[:260]):
            text = clean(item.get("text"))
            if not text:
                continue

            if label_pat.search(text):
                inline = strip_label_prefix(text)
                if inline and looks_like_name(inline) and inline.lower() != text.lower():
                    return make_span(item, inline)
                prev = find_previous_candidate(i)
                if prev:
                    candidate, source = prev
                    return make_span(source, candidate)

            if i + 1 < len(ordered):
                merged = f"{text} {clean(ordered[i + 1].get('text'))}".strip()
                if label_pat.search(merged):
                    if looks_like_name(text):
                        return make_span(item, text)
                    inline = strip_label_prefix(text)
                    if inline and looks_like_name(inline) and inline.lower() != text.lower():
                        return make_span(item, inline)
                    prev = find_previous_candidate(i)
                    if prev:
                        candidate, source = prev
                        return make_span(source, candidate)

        # Second pass: some filings surface the name in the signature block.
        for i, item in enumerate(ordered[-120:]):
            text = clean(item.get("text"))
            if not text:
                continue
            if signature_pat.search(text) or text.lower() == "(registrant)":
                idx = len(ordered) - 120 + i
                prev = find_previous_candidate(idx)
                if prev:
                    candidate, source = prev
                    return make_span(source, candidate)

        # Third pass: news-release exhibits often identify the company in a
        # ticker-tagged lead sentence rather than with an explicit registrant label.
        ticker_pat = re.compile(r"(?i)\[(?:nyse|nasdaq|otc|amex)(?:[^\]]*)\]")
        for item in ordered[:80]:
            text = clean(item.get("text"))
            if not text or "[" not in text:
                continue
            if not ticker_pat.search(text):
                continue
            prefix = clean(text.split("[", 1)[0])
            if not prefix:
                continue
            prefix = re.split(r"[–—-]", prefix)[-1].strip()
            if looks_like_name(prefix):
                return make_span(item, prefix)

        # Fourth pass: use paragraph order as a backup for odd line-breaking.
        para_ordered = sorted(
            [x for x in paragraphs if isinstance(x, dict)],
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("paragraph_no", 10**9),
            ),
        )
        for i, item in enumerate(para_ordered[:180]):
            text = clean(item.get("text"))
            if not text:
                continue
            if label_pat.search(text):
                inline = strip_label_prefix(text)
                if inline and looks_like_name(inline) and inline.lower() != text.lower():
                    return make_span(item, inline)
                for j in range(i - 1, max(-1, i - 4), -1):
                    prev_text = clean(para_ordered[j].get("text"))
                    if prev_text and looks_like_name(prev_text):
                        return make_span(para_ordered[j], prev_text)

        return []
    except Exception:
        return []
