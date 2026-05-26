import re


_HEADER_RE = re.compile(
    r"^\s*Appeals?\s+from\s+the\s+(?:United\s+States\s+)?District\s+Court\b",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"(?i)\b(?:before:|opinion|summary|counsel|argued|submitted|filed|order|"
    r"district judge|magistrate judge|circuit judges?|judge\b|presiding)\b"
)


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

        for i, item in enumerate(lines):
            if i > 120:
                break

            first_text = " ".join((item.get("text") or "").split())
            if not first_text or not _HEADER_RE.search(first_text):
                continue

            collected = [first_text]
            for j in range(i + 1, min(i + 5, len(lines))):
                text = " ".join((lines[j].get("text") or "").split())
                if not text:
                    if collected:
                        break
                    continue
                if _STOP_RE.search(text):
                    break
                collected.append(text)

            span_text = "\n".join(collected).strip()
            if not span_text or span_text in seen:
                continue

            span = {"text": span_text}
            for key in ("page_no", "line_no", "paragraph_no"):
                value = item.get(key)
                if value is not None:
                    span[key] = value
            spans.append(span)
            seen.add(span_text)

        return spans
    except Exception:
        return []
