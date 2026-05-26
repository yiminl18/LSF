import re


_COURT_LINE_RE = re.compile(
    r"(?i)\b(?:for\s+the\s+)?(?:district\s+of|northern\s+district\s+of|"
    r"southern\s+district\s+of|eastern\s+district\s+of|western\s+district\s+of|"
    r"central\s+district\s+of|middle\s+district\s+of)\b"
)
_FRAGMENT_RE = re.compile(
    r"(?i)^(?:for\s+the\s+)?(?:district\s+of|northern\s+district\s+of|"
    r"southern\s+district\s+of|eastern\s+district\s+of|western\s+district\s+of|"
    r"central\s+district\s+of|middle\s+district\s+of)\s*$"
)
_STOP_RE = re.compile(
    r"(?i)\b(?:respondent|petition|order|opinion|before:|argued|submitted|filed|"
    r"summary|counsel|judge|presiding|plaintiffs?\b|defendants?\b|appellants?\b|"
    r"appellees?\b|movant\b|intervenor\b)\b"
)
_STATEISH_RE = re.compile(r"^[A-Z][A-Za-z.&' -]+,?$")


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip()


def rule_district_court_caption_fallback(doc: dict) -> list[dict]:
    try:
        lines = [
            item
            for item in (doc.get("lines") or [])
            if isinstance(item, dict)
        ]
        lines = sorted(
            lines,
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        spans = []
        seen = set()

        def add_span(text: str, source: dict) -> None:
            cleaned = _clean(text).rstrip(" ,;:.")
            key = cleaned.lower()
            if not cleaned or key in seen:
                return
            span = {"text": cleaned}
            for key_name in ("page_no", "line_no", "paragraph_no"):
                value = source.get(key_name)
                if value is not None:
                    span[key_name] = value
            spans.append(span)
            seen.add(key)

        for i, item in enumerate(lines[:120]):
            text = _clean(item.get("text") or "")
            if not text:
                continue
            if "appeal from" in text.lower():
                continue
            if not _COURT_LINE_RE.search(text):
                continue

            prev_window = " ".join(
                _clean(lines[k].get("text") or "")
                for k in range(max(0, i - 6), i)
            ).lower()
            if "d.c. no." not in prev_window and "d.c. nos." not in prev_window:
                continue

            collected = [text]
            if _FRAGMENT_RE.match(text):
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_text = _clean(lines[j].get("text") or "")
                    if not next_text:
                        continue
                    if _STOP_RE.search(next_text):
                        break
                    if _STATEISH_RE.match(next_text):
                        collected.append(next_text)
                    break

            add_span(" ".join(collected), item)

        return spans
    except Exception:
        return []
