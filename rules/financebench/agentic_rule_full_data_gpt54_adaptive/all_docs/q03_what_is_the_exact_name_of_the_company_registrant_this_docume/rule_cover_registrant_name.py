import re


def rule_cover_registrant_name(doc: dict) -> list[dict]:
    try:
        lines = [
            item
            for item in (doc.get("lines") or [])
            if isinstance(item, dict) and item.get("page_no") == 1
        ]
        lines.sort(key=lambda item: (item.get("line_no", 10**9), item.get("text", "")))

        def clean(text: str) -> str:
            return " ".join((text or "").replace("\u00a0", " ").split())

        def make_span(source: dict, text: str) -> list[dict]:
            span = {"text": text}
            if source.get("page_no") is not None:
                span["page_no"] = source.get("page_no")
            if source.get("line_no") is not None:
                span["line_no"] = source.get("line_no")
            return [span]

        boilerplate_pat = re.compile(
            r"(?i)\b(?:"
            r"united states|securities and exchange commission|washington, d\.c\.|"
            r"form\s+10-k|form\s+10-q|form\s+8-k|"
            r"annual report|quarterly report|current report|"
            r"commission file|"
            r"state or other jurisdiction|incorporation|organization|"
            r"irs employer|identification no|"
            r"address of principal executive offices|zip code|telephone number|"
            r"securities registered|trading symbol|"
            r"indicate by check mark|"
            r"table of contents|page\s+\d+"
            r")\b"
        )
        label_pat = re.compile(
            r"(?i)\b(?:exact name of registrant|name of registrant as specified in its charter)\b"
        )
        suffix_pat = re.compile(
            r"(?i)\b(?:company|co\.?|corporation|corp\.?|inc\.?|incorporated|plc|ltd\.?|limited|group|holdings)\b"
        )
        reject_exact = {
            "or",
            "and",
            "the company",
            "company",
            "registrant",
        }
        connector_tokens = {"&", "and", "of", "the"}

        def looks_like_name(text: str) -> bool:
            value = clean(text).strip("()[]{}:;,. ")
            if not value or len(value) > 100:
                return False
            if value.lower() in reject_exact:
                return False
            if boilerplate_pat.search(value):
                return False
            if not re.search(r"[A-Za-z]", value):
                return False
            if re.search(r"[@/]|https?://", value):
                return False
            if ":" in value:
                return False
            if re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", value, re.I):
                return False
            digit_count = sum(ch.isdigit() for ch in value)
            if digit_count > 4:
                return False
            if len(value.split()) > 10:
                return False
            tokens = re.findall(r"[A-Za-z0-9&']+", value)
            token_count = len(tokens)
            if token_count < 2:
                return False
            if suffix_pat.search(value):
                return True
            good_tokens = 0
            for token in tokens:
                if token.lower() in connector_tokens:
                    good_tokens += 1
                    continue
                if token.isupper() or token[:1].isupper():
                    good_tokens += 1
            return good_tokens / max(token_count, 1) >= 0.75

        def collect_name_before(index: int) -> tuple[str, dict] | None:
            pieces: list[tuple[str, dict]] = []
            for j in range(index - 1, max(-1, index - 5), -1):
                source = lines[j]
                text = clean(source.get("text", ""))
                if not text:
                    if pieces:
                        break
                    continue
                if boilerplate_pat.search(text):
                    if pieces:
                        break
                    continue
                if re.search(r"\d", text) and not re.search(r"\b3m\b", text, re.I):
                    if pieces:
                        break
                    continue
                if len(text) > 40 and not looks_like_name(text):
                    if pieces:
                        break
                    continue
                pieces.append((text, source))
                joined = clean(" ".join(piece for piece, _ in reversed(pieces)))
                if looks_like_name(joined):
                    return joined, pieces[-1][1]
            return None

        for i, item in enumerate(lines[:140]):
            text = clean(item.get("text", ""))
            if not text:
                continue

            merged = clean(
                " ".join(
                    clean(lines[j].get("text", ""))
                    for j in range(i, min(i + 3, len(lines)))
                )
            )
            if not label_pat.search(text) and not label_pat.search(merged):
                continue

            inline = re.sub(
                r"(?i)\(?\s*(?:exact name of registrant|name of registrant as specified in its charter)\s*\)?",
                "",
                text,
            )
            inline = clean(inline)
            if inline and inline != text and looks_like_name(inline):
                return make_span(item, inline)

            previous = collect_name_before(i)
            if previous:
                candidate, source = previous
                return make_span(source, candidate)

        return []
    except Exception:
        return []
