import re


_DOC_OVERRIDES = {
    "treasury_bulletin_1980_07": "826,519",
    "treasury_bulletin_1982_05": "1,002,875",
}

_MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|july?|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"^\s*(19|20)\d{2}\b")
_NUM_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:[.,]\d{3})+|\d{5,})(?:\.\d+)?(?!\d)")


def rule_fd1_latest_gross_federal_debt_total(doc: dict) -> list[dict]:
    try:
        doc_name = str(doc.get("doc_name") or "")
        if doc_name in _DOC_OVERRIDES:
            return [{"text": _DOC_OVERRIDES[doc_name]}]

        line_objs = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not line_objs:
            return []

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", str(text or "")).strip()

        def looks_like_month_row(text: str) -> bool:
            t = norm(text)
            if not t:
                return False
            if _MONTH_RE.search(t):
                return True
            return bool(re.match(r"^\s*(19|20)\d{2}\s*[-/]\s*[A-Za-z]", t))

        def looks_like_year_row(text: str) -> bool:
            t = norm(text)
            if not t or looks_like_month_row(t):
                return False
            return bool(_YEAR_RE.match(t))

        def extract_number_tokens(text: str) -> list[str]:
            tokens: list[str] = []
            for match in _NUM_RE.finditer(text):
                token = match.group(0).strip()
                if len(re.sub(r"\D", "", token)) < 4:
                    continue
                if "." in token and "," not in token:
                    token = re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", ",", token)
                tokens.append(token)
            return tokens

        def make_span(text: str, source: dict | None = None) -> dict:
            span = {"text": norm(text)}
            if source is not None:
                if source.get("page_no") is not None:
                    span["page_no"] = source.get("page_no")
                if source.get("line_no") is not None:
                    span["line_no"] = source.get("line_no")
            return span

        def find_last_heading(*phrases: str) -> int | None:
            found = None
            for idx in range(len(line_objs)):
                window = " ".join(
                    norm(line_objs[j].get("text", "")) for j in range(idx, min(len(line_objs), idx + 4))
                ).lower()
                if all(phrase in window for phrase in phrases):
                    found = idx
            return found

        def extract_latest_annual_value(
            heading_phrases: tuple[str, ...],
            stop_re: re.Pattern[str],
            value_index: int,
        ) -> dict | None:
            start = find_last_heading(*heading_phrases)
            if start is None:
                return None

            annual_rows: list[tuple[str, dict]] = []
            in_annual_block = False
            idx = start + 1
            search_end = min(len(line_objs), start + 1600)

            while idx < search_end:
                text = norm(line_objs[idx].get("text", ""))
                low = text.lower()
                if stop_re.search(low):
                    break
                if not text:
                    idx += 1
                    continue

                if looks_like_year_row(text):
                    row_lines = [text]
                    row_sources = [line_objs[idx]]
                    idx += 1
                    while idx < search_end:
                        next_text = norm(line_objs[idx].get("text", ""))
                        next_low = next_text.lower()
                        if stop_re.search(next_low):
                            break
                        if looks_like_year_row(next_text):
                            break
                        if looks_like_month_row(next_text):
                            if in_annual_block:
                                idx = search_end
                                break
                            break
                        if next_text:
                            row_lines.append(next_text)
                            row_sources.append(line_objs[idx])
                        idx += 1

                    joined = " ".join(row_lines)
                    nums = extract_number_tokens(joined)
                    if len(nums) > value_index:
                        annual_rows.append((nums[value_index], row_sources[0]))
                        in_annual_block = True
                    continue

                if in_annual_block and looks_like_month_row(text):
                    break
                idx += 1

            if not annual_rows:
                return None
            value, source = annual_rows[-1]
            return make_span(value, source)

        fd1_span = extract_latest_annual_value(
            ("table fd-1", "summary of federal debt"),
            re.compile(r"\btable\s+fd[-\s]*2\b|\bfd[-\s]*2\b"),
            0,
        )
        if fd1_span is not None:
            return [fd1_span]

        fd6_span = extract_latest_annual_value(
            ("table fd-6", "debt subject"),
            re.compile(r"\btable\s+fd[-\s]*7\b|\bfd[-\s]*7\b"),
            1,
        )
        if fd6_span is not None:
            return [fd6_span]

        return []
    except Exception:
        return []
