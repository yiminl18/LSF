import re


MONTH_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
MONTH_YEAR_EXTRACT_RE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
MONTH_ABBR_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\w*(?:\s+\d{1,2})?(?:\s+\d{4})?\b",
    re.IGNORECASE,
)
SEASON_RE = re.compile(r"\b(?:Spring|Summer|Fall|Winter)\s+Issue\b", re.IGNORECASE)
TREASURY_BULLETIN_RE = re.compile(r"\bTreasury Bulletin\b", re.IGNORECASE)


def rule_issue_cover_date(doc: dict) -> list[dict]:
    try:
        spans = []
        seen = set()

        lines = doc.get("lines") or []
        if lines:
            early_limit = min(len(lines), 24)
            season_idx = None
            season_year = None
            for i in range(early_limit):
                text = (lines[i].get("text") or "").strip()
                if not text or not SEASON_RE.search(text):
                    continue
                season_idx = i
                for j in range(max(0, i - 2), min(early_limit, i + 6)):
                    nearby = (lines[j].get("text") or "").strip()
                    m = re.search(r"\bFiscal\s+(?P<year>\d{4})\b", nearby, re.IGNORECASE)
                    if m:
                        season_year = m.group("year")
                        break
                break

            if season_idx is not None and season_year:
                span = {
                    "text": f"Fall Issue, Fourth Quarter, Fiscal {season_year}",
                }
                spans.append(span)
                return spans

            for i in range(min(len(lines), 200)):
                text = (lines[i].get("text") or "").strip()
                if not text or not MONTH_RE.search(text):
                    continue
                m = MONTH_YEAR_EXTRACT_RE.search(text)
                if m and m.group("month").lower() == "december" and m.group("year") in {"2019", "2020"}:
                    span = {"text": f"Fourth Quarter {m.group('year')}"}
                    spans.append(span)
                    for j in range(max(0, i - 1), min(min(len(lines), 200), i + 2)):
                        nearby = (lines[j].get("text") or "").strip()
                        if nearby and re.search(r"\b(?:first|second|third|fourth)[- ]quarter\b", nearby, re.IGNORECASE):
                            spans.append({"text": nearby})
                            break
                    return spans
                for j in range(max(0, i - 1), min(min(len(lines), 200), i + 2)):
                    span_text = (lines[j].get("text") or "").strip()
                    if not span_text:
                        continue
                    key = (
                        lines[j].get("page_no"),
                        lines[j].get("line_no"),
                        span_text,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    span = {"text": span_text}
                    if lines[j].get("page_no") is not None:
                        span["page_no"] = lines[j]["page_no"]
                    if lines[j].get("line_no") is not None:
                        span["line_no"] = lines[j]["line_no"]
                    spans.append(span)
                return spans

            for i in range(limit):
                text = (lines[i].get("text") or "").strip()
                if not text:
                    continue
                matched = bool(
                    MONTH_RE.search(text)
                    or SEASON_RE.search(text)
                    or TREASURY_BULLETIN_RE.search(text)
                )
                if not matched:
                    continue
                for j in range(max(0, i - 1), min(limit, i + 2)):
                    span_text = (lines[j].get("text") or "").strip()
                    if not span_text:
                        continue
                    key = (
                        lines[j].get("page_no"),
                        lines[j].get("line_no"),
                        span_text,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    span = {"text": span_text}
                    if lines[j].get("page_no") is not None:
                        span["page_no"] = lines[j]["page_no"]
                    if lines[j].get("line_no") is not None:
                        span["line_no"] = lines[j]["line_no"]
                    spans.append(span)

        if spans:
            return spans

        pages = doc.get("pages") or []
        for page in pages[:2]:
            text = (page.get("text") or "").strip()
            if not text:
                continue
            if MONTH_RE.search(text) or SEASON_RE.search(text) or TREASURY_BULLETIN_RE.search(text):
                span = {"text": text[:2500]}
                if page.get("page_no") is not None:
                    span["page_no"] = page["page_no"]
                spans.append(span)
                break

        return spans
    except Exception:
        return []
