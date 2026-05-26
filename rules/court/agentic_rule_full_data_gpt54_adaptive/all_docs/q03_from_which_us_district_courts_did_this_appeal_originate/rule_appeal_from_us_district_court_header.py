import re


_HEADER_RE = re.compile(
    r"^\s*Appeals?\s+from\s+the\s+(?:United\s+States\s+)?District\s+Court\b",
    re.IGNORECASE,
)
_DISTRICT_RE = re.compile(
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
    r"(?i)\b(?:before:|opinion|summary|counsel|argued|submitted|filed|order|"
    r"district judge|magistrate judge|circuit judges?|judge\b|presiding)\b"
)


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip().rstrip(" ,;:.")


def rule_appeal_from_us_district_court_header(doc: dict) -> list[dict]:
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

        for i, item in enumerate(lines[:300]):
            first_text = _clean(item.get("text") or "")
            if not first_text or not _HEADER_RE.search(first_text):
                continue

            span_lines = [first_text]
            if not _DISTRICT_RE.search(first_text):
                for j in range(i + 1, min(i + 8, len(lines))):
                    next_text = _clean(lines[j].get("text") or "")
                    if not next_text:
                        continue
                    if _STOP_RE.search(next_text):
                        break
                    if _DISTRICT_RE.search(next_text) or next_text.lower().startswith("for the "):
                        span_lines.append(next_text)
                        if _FRAGMENT_RE.match(next_text):
                            for k in range(j + 1, min(j + 3, len(lines))):
                                tail_text = _clean(lines[k].get("text") or "")
                                if not tail_text:
                                    continue
                                if _STOP_RE.search(tail_text):
                                    break
                                span_lines.append(tail_text)
                                break
                        break

            span_text = "\n".join(span_lines).strip()
            key = span_text.lower()
            if not span_text or key in seen:
                continue

            span = {"text": span_text}
            for key_name in ("page_no", "line_no", "paragraph_no"):
                value = item.get(key_name)
                if value is not None:
                    span[key_name] = value
            spans.append(span)
            seen.add(key)

        return spans
    except Exception:
        return []
