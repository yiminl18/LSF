import re


_SUMMARY_RE = re.compile(r"^\s*SUMMARY(?:\s*\*+)?\s*(.*)$", re.IGNORECASE)
_BODY_START_RE = re.compile(
    r"^(?:"
    r"The panel\b|"
    r"The en banc court\b|"
    r"The court\b|"
    r"The district court\b|"
    r"Affirming\b|"
    r"Reversing\b|"
    r"Vacating\b|"
    r"Remanding\b|"
    r"Dismissing\b|"
    r"Granting\b|"
    r"Denying\b|"
    r"Holding\b|"
    r"In (?:a|an|this|these|that)\b|"
    r"On (?:appeal|remand)\b|"
    r"Because\b|"
    r"We\b|"
    r"Petitioners?\b|"
    r"Plaintiffs?\b|"
    r"Defendants?\b|"
    r"This summary\b|"
    r"This case\b|"
    r"This appeal\b"
    r")",
    re.IGNORECASE,
)
_DISCLAIMER_RE = re.compile(r"^This summary constitutes no part", re.IGNORECASE)
_LOWER_OK = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "de",
    "del",
    "du",
    "for",
    "from",
    "in",
    "la",
    "law",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "under",
    "v",
    "vs",
    "with",
    "without",
}


def rule_summary_topic(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", text or "").strip()

        def clean(text: str) -> str:
            text = norm(text)
            text = re.sub(r"^[\[\](){}*\s]+", "", text)
            text = re.sub(r"[\[\](){}*\s]+$", "", text)
            return text.strip(" -:;,")

        def is_captionish(text: str) -> bool:
            if not text:
                return True
            if re.fullmatch(r"\d+", text):
                return True
            if re.fullmatch(r"\*+", text):
                return True
            return (
                len(text) >= 12
                and text == text.upper()
                and sum(ch.isalpha() for ch in text) >= 8
            )

        def is_heading(text: str) -> bool:
            if not text or len(text) > 95:
                return False
            if text.endswith("."):
                return False
            if _DISCLAIMER_RE.match(text) or _BODY_START_RE.match(text) or is_captionish(text):
                return False
            if not re.search(r"[A-Za-z]", text):
                return False

            tokens = re.findall(r"[A-Za-z][A-Za-z'’.-]*|\d+", text)
            if not tokens:
                return False

            good = 0
            bad = 0
            for token in tokens:
                lowered = token.lower().strip(".")
                if token.isdigit() or lowered in _LOWER_OK or token[0].isupper() or token.isupper():
                    good += 1
                else:
                    bad += 1

            return bad <= max(1, len(tokens) // 4) and good >= bad + 1

        def ordered(items: list[dict]) -> list[dict]:
            def sort_key(item: dict) -> tuple[int, int, int]:
                return (
                    item.get("page_no") if isinstance(item.get("page_no"), int) else 0,
                    item.get("line_no") if isinstance(item.get("line_no"), int) else 0,
                    item.get("paragraph_no") if isinstance(item.get("paragraph_no"), int) else 0,
                )

            return sorted([item for item in items if isinstance(item, dict)], key=sort_key)

        def with_meta(text: str, item: dict) -> dict:
            span = {"text": text}
            for key in ("page_no", "line_no", "paragraph_no"):
                if item.get(key) is not None:
                    span[key] = item[key]
            return span

        def extract(items: list[dict]) -> list[dict]:
            data = ordered(items)
            for idx, item in enumerate(data):
                raw = norm(item.get("text"))
                match = _SUMMARY_RE.match(raw)
                if not match:
                    continue

                parts: list[str] = []
                source_item = None

                inline = clean(match.group(1))
                if inline and is_heading(inline):
                    parts.append(inline)
                    source_item = item

                for follower in data[idx + 1 : idx + 8]:
                    candidate = clean(follower.get("text"))
                    if not candidate:
                        if parts:
                            break
                        continue
                    if _DISCLAIMER_RE.match(candidate) or _BODY_START_RE.match(candidate):
                        if parts:
                            return [
                                with_meta(" ".join(parts), source_item),
                                with_meta(candidate, follower),
                            ]
                        break
                    if not is_heading(candidate):
                        if parts:
                            return [
                                with_meta(" ".join(parts), source_item),
                                with_meta(candidate, follower),
                            ]
                        break
                    if source_item is None:
                        source_item = follower
                    parts.append(candidate)
                    if len(parts) >= 3:
                        break

                if parts and source_item is not None:
                    return [with_meta(" ".join(parts), source_item)]

            return []

        for key in ("lines", "paragraphs", "pages"):
            found = extract(doc.get(key) or [])
            if found:
                return found

        return []
    except Exception:
        return []
