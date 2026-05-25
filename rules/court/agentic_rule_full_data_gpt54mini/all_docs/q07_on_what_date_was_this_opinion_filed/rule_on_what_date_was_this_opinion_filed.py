import re


_MONTH_NAME_RE = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)"
)
_DATE_LONG_RE = re.compile(rf"^{_MONTH_NAME_RE} \d{{1,2}}, \d{{4}}$")
_DATE_ABBR_RE = re.compile(r"^[A-Z]{3} \d{1,2} \d{4}$")
_FILED_LONG_RE = re.compile(rf"^Filed ({_MONTH_NAME_RE} \d{{1,2}}, \d{{4}})$")
_FILED_ABBR_RE = re.compile(r"^Filed ([A-Z]{3} \d{1,2} \d{4})$")


def _as_int(value):
    try:
        return int(value)
    except Exception:
        return value


def _iter_header_lines(doc, max_pages=12, max_lines=100):
    pages = doc.get("pages") or []
    if pages:
        pages = sorted(pages, key=lambda x: _as_int(x.get("page_no", 0)))
        for page in pages[:max_pages]:
            page_no = _as_int(page.get("page_no", 0))
            text = page.get("text") or ""
            for idx, raw_line in enumerate(text.splitlines()[:max_lines], start=1):
                yield {"page_no": page_no, "line_no": idx, "text": raw_line}
        return

    lines = doc.get("lines") or []
    if lines:
        lines = sorted(
            lines,
            key=lambda x: (
                _as_int(x.get("page_no", 0)),
                _as_int(x.get("line_no", 0)),
            ),
        )
        current_page = None
        seen_pages = 0
        for line in lines:
            page_no = _as_int(line.get("page_no", 0))
            if page_no != current_page:
                current_page = page_no
                seen_pages += 1
                if seen_pages > max_pages:
                    break
            if _as_int(line.get("line_no", 0)) > max_lines:
                continue
            yield line


def _make_span(line, text=None):
    span = {
        "text": line.get("text", "") if text is None else text,
    }
    if "page_no" in line:
        span["page_no"] = line["page_no"]
    if "line_no" in line:
        span["line_no"] = line["line_no"]
    return span


def rule_on_what_date_was_this_opinion_filed(doc: dict) -> list[dict]:
    try:
        full_text = doc.get("text") or ""

        # Withdrawn-opinion orders often say "The Opinion filed <date> ..." and
        # the question is asking about that underlying opinion, not the order itself.
        m = re.search(
            rf"\bThe Opinion filed ({_MONTH_NAME_RE} \d{{1,2}}, \d{{4}}|[A-Z]{{3}} \d{{1,2}} \d{{4}})",
            full_text,
        )
        if m and (re.search(r"\bFILED\b", full_text) or re.search(r"\bAMENDED\b", full_text)):
            return [{"text": m.group(1)}]

        header_lines = list(_iter_header_lines(doc))
        if not header_lines:
            return []

        # The filing date usually appears on page 1 or 2, near the top of the header.
        for idx, line in enumerate(header_lines):
            text = (line.get("text") or "").strip()
            if not text:
                continue

            m = _FILED_LONG_RE.match(text) or _FILED_ABBR_RE.match(text)
            if m:
                return [_make_span(line, m.group(1))]

            if text == "FILED":
                for follow in header_lines[idx + 1 :]:
                    next_text = (follow.get("text") or "").strip()
                    if not next_text:
                        continue
                    if _DATE_LONG_RE.match(next_text) or _DATE_ABBR_RE.match(next_text):
                        return [_make_span(follow)]
                    break

            if _DATE_LONG_RE.match(text) or _DATE_ABBR_RE.match(text):
                return [_make_span(line)]

        # Fallback for unusual layouts.
        for pat in (_FILED_LONG_RE, _FILED_ABBR_RE):
            m = pat.search(full_text)
            if m:
                return [{"text": m.group(1)}]

        m = re.search(rf"\n({_MONTH_NAME_RE} \d{{1,2}}, \d{{4}}|[A-Z]{{3}} \d{{1,2}} \d{{4}})\n", full_text)
        if m:
            return [{"text": m.group(1)}]

        return []
    except Exception:
        return []
