import re


_COURT_LINE_RE = re.compile(
    r"(?i)\b(?:united states district court|u\.s\. district court|"
    r"(?:north|south|east|west|central|middle)ern?\s+district\s+of\b|"
    r"district of\b)"
)
_TITLE_STOP_RE = re.compile(
    r"(?i)\b(?:respondent|petition|order|opinion|before:|argued|submitted|filed|"
    r"summary|counsel|judge|presiding)\b"
)
_CITY_CONT_RE = re.compile(r"^[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)*(?:,)?$")


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
            cleaned = " ".join(text.split()).strip()
            if not cleaned or cleaned in seen:
                return
            span = {"text": cleaned}
            for key in ("page_no", "line_no", "paragraph_no"):
                value = source.get(key)
                if value is not None:
                    span[key] = value
            spans.append(span)
            seen.add(cleaned)

        for i, item in enumerate(lines[:90]):
            text = " ".join((item.get("text") or "").split())
            if not text:
                continue

            if "appeal from" in text.lower():
                continue

            if _COURT_LINE_RE.search(text):
                collected = [text]
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_text = " ".join((lines[j].get("text") or "").split())
                    if not next_text:
                        if collected:
                            break
                        continue
                    if _TITLE_STOP_RE.search(next_text):
                        break
                    collected.append(next_text)
                add_span("\n".join(collected), item)
                continue

            prev_window = " ".join(
                " ".join((lines[k].get("text") or "").split())
                for k in range(max(0, i - 4), i)
            ).lower()
            if "d.c. no." not in prev_window and "no." not in prev_window:
                continue

            if not re.search(
                r"(?i)\b(?:northern|southern|eastern|western|central|middle)\s+district\s+of\b|\bdistrict\s+of\b",
                text,
            ):
                continue

            collected = [text]
            for j in range(i + 1, min(i + 4, len(lines))):
                next_text = " ".join((lines[j].get("text") or "").split())
                if not next_text:
                    if collected:
                        break
                    continue
                if _TITLE_STOP_RE.search(next_text):
                    break
                if _CITY_CONT_RE.match(next_text) or _COURT_LINE_RE.search(next_text):
                    collected.append(next_text)
                    continue
                break
            add_span("\n".join(collected), item)

        return spans
    except Exception:
        return []
