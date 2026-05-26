import re


_MONTH_RE = re.compile(
    r"\b("
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|"
    r"october|november|december"
    r")\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"^\s*(?:19|20)\d{2}\b")


def _normalize(text: str) -> str:
    text = str(text or "")
    for old, new in (("—", "-"), ("–", "-"), ("−", "-"), ("•", " ")):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize(text).lower())


def _window(lines: list[dict], start: int, width: int) -> str:
    return " ".join(_normalize(lines[i].get("text") or "") for i in range(start, min(len(lines), start + width)))


def _line_is_year(text: str) -> bool:
    cleaned = _normalize(text)
    if not cleaned or not _YEAR_RE.match(cleaned):
        return False
    if _MONTH_RE.search(cleaned):
        return False
    low = cleaned.lower()
    return "t.q" not in low and "t.o" not in low


def _line_is_month_or_partial(text: str) -> bool:
    cleaned = _normalize(text)
    if not cleaned:
        return False
    low = cleaned.lower()
    if _MONTH_RE.search(cleaned):
        return True
    if re.match(r"^(?:19|20)\d{2}\s*-\s*[a-z]", low):
        return True
    return low in {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


def _year_value(text: str) -> int | None:
    m = _YEAR_RE.match(_normalize(text))
    if not m:
        return None
    try:
        return int(m.group(0).strip()[:4])
    except Exception:
        return None


def _is_question_anchor(window_compact: str) -> bool:
    has_title = "maturitydistributionandaveragelength" in window_compact and (
        "marketableinterestbearingpublicdebt" in window_compact
        or ("marketable" in window_compact and "publicdebt" in window_compact)
        or "marketabledebt" in window_compact
    )
    has_holder = "heldbyprivateinvestors" in window_compact or "privateinvestors" in window_compact or "privatelyheld" in window_compact
    has_footnote = "september1976" in window_compact and "averagelength" in window_compact and (
        "privateinvestors" in window_compact or "privatelyheld" in window_compact
    )
    return (has_title and has_holder) or has_footnote


def _find_strict_title_idx(lines: list[dict]) -> int | None:
    for idx in range(len(lines)):
        window_compact = _compact(_window(lines, idx, 12))
        if not (
            "maturitydistributionandaveragelength" in window_compact
            and ("privateinvestors" in window_compact or "privatelyheld" in window_compact)
            and ("marketable" in window_compact or "publicdebt" in window_compact)
        ):
            continue
        tail = _compact(_window(lines, idx + 1, 30))
        if "endoffiscalyear" in tail or "fiscalyearormonth" in tail:
            return idx
    return None


def _find_chart_bounds(lines: list[dict]) -> tuple[int, int] | None:
    for idx in range(len(lines)):
        window_compact = _compact(_window(lines, idx, 8))
        if "chartsfda" not in window_compact or "averagelength" not in window_compact:
            continue
        start = idx
        for j in range(idx, min(len(lines), idx + 8)):
            if "chartsfda" in _compact(lines[j].get("text") or ""):
                start = j
                break
        tail = _compact(_window(lines, start, 16))
        if "privatelyheld" not in tail and "dec31" not in tail and "years" not in tail:
            continue
        end = min(len(lines), start + 24)
        for j in range(start + 4, min(len(lines), start + 28)):
            compact = _compact(_window(lines, j, 4))
            if "tablefd" in compact or "chartsfd" in compact:
                end = j
                break
        return start, end
    return None


def _extract_chart_answer_bounds(lines: list[dict], start: int, end: int) -> tuple[int, int] | None:
    for idx in range(start, end):
        text = _normalize(lines[idx].get("text") or "")
        low = text.lower()
        if not text or not any(ch.isdigit() for ch in text):
            continue
        if "dec" not in low and "sept" not in low and "sep" not in low:
            continue
        for j in range(idx + 1, min(end, idx + 4)):
            nxt = _normalize(lines[j].get("text") or "")
            compact = _compact(nxt)
            if "months" in nxt.lower() or "mos" in compact:
                return start, j + 1
    return None


def _score_anchor(lines: list[dict], idx: int) -> int:
    window_compact = _compact(_window(lines, max(0, idx - 2), 14))
    if not _is_question_anchor(window_compact):
        return -1

    nearby = []
    start = max(0, idx - 260)
    end = min(len(lines), idx + 320)
    for j in range(start, end):
        if _line_is_year(lines[j].get("text") or ""):
            nearby.append(j)

    if not nearby:
        return 0

    years = [_year_value(lines[j].get("text") or "") for j in nearby]
    years = [y for y in years if y is not None]
    score = len(nearby)
    if years:
        score += max(years) - min(years)
        score += max(years)

    if "tablefd5" in window_compact or "tablefd7" in window_compact:
        score += 25
    if "endoffiscalyearormonth" in _compact(_window(lines, idx, 24)):
        score += 25
    if "averagelength" in window_compact:
        score += 10
    return score


def _find_best_anchor(lines: list[dict]) -> int | None:
    best_idx = None
    best_score = -1
    for idx in range(len(lines)):
        score = _score_anchor(lines, idx)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx if best_score >= 0 else None


def _find_best_cluster(lines: list[dict], anchor_idx: int) -> list[int]:
    start = max(0, anchor_idx - 260)
    end = min(len(lines), anchor_idx + 340)
    row_idxs = [j for j in range(start, end) if _line_is_year(lines[j].get("text") or "")]
    if not row_idxs:
        return []

    clusters = []
    current = [row_idxs[0]]
    for j in row_idxs[1:]:
        if j - current[-1] <= 35:
            current.append(j)
        else:
            clusters.append(current)
            current = [j]
    clusters.append(current)

    def cluster_key(cluster: list[int]) -> tuple[int, int, int]:
        vals = [_year_value(lines[j].get("text") or "") for j in cluster]
        vals = [v for v in vals if v is not None]
        max_year = max(vals) if vals else -1
        distance = min(abs(anchor_idx - j) for j in cluster)
        return (max_year, len(cluster), -distance)

    return max(clusters, key=cluster_key)


def _find_block_bounds(lines: list[dict], anchor_idx: int, cluster: list[int]) -> tuple[int, int]:
    if not cluster:
        return max(0, anchor_idx - 40), min(len(lines), anchor_idx + 220)

    cluster_start = cluster[0]
    cluster_end = cluster[-1]
    start = max(0, min(anchor_idx, cluster_start) - 40)
    for j in range(min(anchor_idx, cluster_start), max(-1, min(anchor_idx, cluster_start) - 120), -1):
        compact = _compact(_window(lines, max(0, j - 1), 6))
        if _is_question_anchor(compact) or "tablefd5" in compact or "tablefd7" in compact:
            start = j
            break

    end = min(len(lines), max(anchor_idx, cluster_end) + 220)
    scan_start = min(len(lines), max(anchor_idx, cluster_end) + 30)
    for j in range(scan_start, min(len(lines), max(anchor_idx, cluster_end) + 320)):
        compact = _compact(_window(lines, j, 6))
        if ("tablefd" in compact or "chartsfda" in compact or "tablepdo" in compact) and j > max(anchor_idx, cluster_end) + 20:
            end = j
            break

    return start, end


def _find_title_block_end(lines: list[dict], title_idx: int) -> int:
    end = min(len(lines), title_idx + 220)
    for j in range(title_idx + 20, min(len(lines), title_idx + 320)):
        compact = _compact(_window(lines, j, 6))
        if ("tablefd6" in compact or "tablefd7" in compact or "chartsfda" in compact or "tablepdo" in compact):
            end = j
            break
    return end


def _build_span(lines: list[dict], start: int, end: int) -> dict | None:
    parts = []
    for idx in range(start, min(len(lines), end)):
        text = _normalize(lines[idx].get("text") or "")
        if text:
            parts.append(text)
    if not parts:
        return None
    span = {"text": "\n".join(parts)}
    first = lines[start]
    if first.get("page_no") is not None:
        span["page_no"] = first.get("page_no")
    if first.get("line_no") is not None:
        span["line_no"] = first.get("line_no")
    return span


def _build_chart_value_span(lines: list[dict], start: int, end: int) -> dict | None:
    pieces = [_normalize(lines[idx].get("text") or "") for idx in range(start, end)]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return None

    date_line = None
    value_line = None
    for piece in pieces:
        low = piece.lower()
        if any(ch.isdigit() for ch in piece) and ("dec" in low or "sept" in low or "sep" in low):
            date_line = piece
        if "months" in low or "mos" in _compact(piece):
            value_line = piece

    if date_line and value_line:
        text = f"Average length of marketable debt privately held at {date_line}: {value_line}"
    else:
        text = "\n".join(pieces)

    span = {"text": text}
    first = lines[start]
    if first.get("page_no") is not None:
        span["page_no"] = first.get("page_no")
    if first.get("line_no") is not None:
        span["line_no"] = first.get("line_no")
    return span


def _extract_latest_row_bounds(lines: list[dict], start: int, end: int) -> tuple[int, int] | None:
    annual_rows = [idx for idx in range(start, end) if _line_is_year(lines[idx].get("text") or "")]
    if not annual_rows:
        return None

    latest_idx = max(annual_rows, key=lambda idx: (_year_value(lines[idx].get("text") or "") or -1, idx))
    row_end = min(end, latest_idx + 1)
    saw_payload = False

    for idx in range(latest_idx + 1, end):
        text = _normalize(lines[idx].get("text") or "")
        compact = _compact(text)
        if not text:
            continue
        if idx > latest_idx + 1 and (_line_is_year(text) or _line_is_month_or_partial(text)):
            break
        if "tablefd" in compact or "chartsfda" in compact or "tablepdo" in compact:
            break
        row_end = idx + 1
        if any(ch.isdigit() for ch in text) or "yrs" in compact or "mos" in compact or "months" in compact:
            saw_payload = True

    if not saw_payload:
        return None
    return latest_idx, row_end


def rule_marketable_debt_average_length(doc: dict) -> list[dict]:
    try:
        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        ordered = sorted(
            lines,
            key=lambda item: (
                item.get("page_no", 10**9),
                item.get("line_no", 10**9),
            ),
        )

        strict_title_idx = _find_strict_title_idx(ordered)
        if strict_title_idx is not None:
            strict_end = _find_title_block_end(ordered, strict_title_idx)
            row_bounds = _extract_latest_row_bounds(ordered, strict_title_idx, strict_end)
            if row_bounds is not None:
                row_span = _build_span(ordered, row_bounds[0], row_bounds[1])
                if row_span is not None:
                    return [row_span]

        chart_bounds = _find_chart_bounds(ordered)
        if chart_bounds is not None:
            focused_chart_bounds = _extract_chart_answer_bounds(ordered, chart_bounds[0], chart_bounds[1]) or chart_bounds
            chart_span = _build_chart_value_span(ordered, focused_chart_bounds[0], focused_chart_bounds[1])
            if chart_span is not None:
                return [chart_span]

        anchor_idx = _find_best_anchor(ordered)
        if anchor_idx is None:
            return []

        cluster = _find_best_cluster(ordered, anchor_idx)
        start, end = _find_block_bounds(ordered, anchor_idx, cluster)
        row_bounds = _extract_latest_row_bounds(ordered, start, end)
        if row_bounds is not None:
            row_span = _build_span(ordered, row_bounds[0], row_bounds[1])
            if row_span is not None:
                return [row_span]

        block_span = _build_span(ordered, start, end)
        return [block_span] if block_span else []
    except Exception:
        return []
