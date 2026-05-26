import re


_MONTH_TOKENS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

_YEAR_START_RE = re.compile(r"^\s*(?:19|20)\d{2}\b")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _looks_like_year_row(text: str) -> bool:
    low = _normalize(text).lower()
    if not low or not _YEAR_START_RE.match(low):
        return False
    if any(token in low for token in _MONTH_TOKENS):
        return False
    return True


def _looks_like_any_row(text: str) -> bool:
    return bool(_YEAR_START_RE.match(_normalize(text)))


def _extract_year(text: str) -> int | None:
    m = _YEAR_START_RE.match(_normalize(text))
    if not m:
        return None
    try:
        return int(m.group(0).strip()[:4])
    except Exception:
        return None


def _window_text(lines: list[dict], start: int, width: int = 4) -> str:
    parts = []
    for idx in range(start, min(len(lines), start + width)):
        txt = _normalize(lines[idx].get("text") or "")
        if txt:
            parts.append(txt)
    return " ".join(parts)


def _find_table_title_idx(lines: list[dict]) -> int | None:
    for idx in range(len(lines)):
        window = _window_text(lines, idx, width=12).lower()
        if (
            "maturity distribution and average length" in window
            and "marketable interest-bearing public debt" in window
            and "private investors" in window
        ):
            tail = _window_text(lines, idx + 1, width=25).lower()
            block = f"{window} {tail}"
            if "source:" in block and "fiscal year" in block and "average length" in block:
                return idx
    return None


def _find_header_idx(lines: list[dict], start_idx: int = 0) -> int | None:
    for idx in range(max(0, start_idx), min(len(lines), start_idx + 80)):
        window = _window_text(lines, idx, width=8).lower()
        if "end of fiscal" in window or "fiscal year or month" in window:
            return idx
    for idx in range(max(0, start_idx), min(len(lines), start_idx + 80)):
        window = _window_text(lines, idx, width=8).lower()
        if "average length" in window:
            return idx
    return None


def _find_intro_span(lines: list[dict], header_idx: int) -> dict | None:
    target_bits = (
        "table fd-5 illustrates the average length",
        "average length of marketable interest-bearing public debt held by private investors",
        "maturity distribution of that debt",
    )
    for idx in range(max(0, header_idx - 120), min(len(lines), header_idx + 25)):
        window = _window_text(lines, idx, width=4).lower()
        if all(bit in window for bit in target_bits):
            start = idx
            end = min(len(lines), idx + 4)
            while end < len(lines):
                txt = _normalize(lines[end].get("text") or "")
                if not txt:
                    end += 1
                    continue
                if _looks_like_any_row(txt) or txt.lower().startswith("table fd-"):
                    break
                if "average length" in txt.lower() and "months" in txt.lower():
                    break
                end += 1
            snippet = "\n".join(
                _normalize(lines[j].get("text") or "")
                for j in range(start, end)
                if _normalize(lines[j].get("text") or "")
            )
            if snippet:
                return {
                    "text": snippet,
                    "page_no": lines[start].get("page_no"),
                    "line_no": lines[start].get("line_no"),
                }
    return None


def _find_latest_fiscal_year_row(lines: list[dict], header_idx: int) -> dict | None:
    search_end = min(len(lines), header_idx + 250)
    row_starts = []
    for idx in range(header_idx + 1, search_end):
        txt = _normalize(lines[idx].get("text") or "")
        if not txt:
            continue
        if _looks_like_any_row(txt):
            row_starts.append(idx)

    if not row_starts:
        return None

    clusters = []
    current = [row_starts[0]]
    for idx in row_starts[1:]:
        if idx - current[-1] <= 30:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    def _cluster_score(cluster: list[int]) -> int:
        years = [
            y
            for y in (
                _extract_year(lines[idx].get("text") or "")
                for idx in cluster
            )
            if y is not None
        ]
        return max(years) if years else -1

    cluster = max(clusters, key=_cluster_score)
    fiscal_row_starts = [idx for idx in cluster if _looks_like_year_row(lines[idx].get("text") or "")]
    if fiscal_row_starts:
        row_start = fiscal_row_starts[-1]
    else:
        row_start = cluster[-1]

    next_row_candidates = [idx for idx in row_starts if idx > row_start]
    row_end = next_row_candidates[0] if next_row_candidates else search_end

    snippet_lines = []
    for idx in range(row_start, row_end):
        txt = _normalize(lines[idx].get("text") or "")
        if txt:
            snippet_lines.append(txt)

    if not snippet_lines:
        return None

    return {
        "text": "\n".join(snippet_lines),
        "page_no": lines[row_start].get("page_no"),
        "line_no": lines[row_start].get("line_no"),
    }


def rule_marketable_debt_average_length(doc: dict) -> list[dict]:
    try:
        lines = doc.get("lines") or []
        ordered = sorted(
            lines,
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        if not ordered:
            return []

        title_idx = _find_table_title_idx(ordered)
        if title_idx is None:
            return []

        header_idx = _find_header_idx(ordered, title_idx + 1)
        if header_idx is None:
            return []

        spans = []

        title_end = min(len(ordered), header_idx)
        title_snippet = "\n".join(
            _normalize(ordered[j].get("text") or "")
            for j in range(title_idx, title_end)
            if _normalize(ordered[j].get("text") or "")
        )
        if title_snippet:
            spans.append(
                {
                    "text": title_snippet,
                    "page_no": ordered[title_idx].get("page_no"),
                    "line_no": ordered[title_idx].get("line_no"),
                }
            )

        latest_row = _find_latest_fiscal_year_row(ordered, header_idx)
        if latest_row:
            spans.append(latest_row)

        return spans
    except Exception:
        return []
