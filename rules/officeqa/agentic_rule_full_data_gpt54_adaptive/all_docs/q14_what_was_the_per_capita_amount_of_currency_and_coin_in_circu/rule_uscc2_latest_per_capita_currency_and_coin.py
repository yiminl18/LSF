import re


DATE_RE = re.compile(
    r"(?:"
    r"Jan(?:uary)?\.?|Feb(?:ruary)?\.?|Mar(?:ch)?\.?|Apr(?:il)?\.?|"
    r"May|Jun(?:e)?\.?|Jul(?:y)?\.?|Aug(?:ust)?\.?|"
    r"Sep(?:t(?:ember)?)?\.?|Sept\.?|Oct(?:ober)?\.?|"
    r"Nov(?:ember)?\.?|Dec(?:ember)?\.?"
    r")\s+\d{1,2},\s+\d{4}\b",
    re.IGNORECASE,
)
MONTH_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z.]*\b",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"^\s*[$rR]?\s*\d[\d,.\s]*\s*$")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(text).lower())


def _window_text(lines: list[str], start: int, end: int) -> str:
    parts = []
    for idx in range(max(0, start), min(len(lines), end)):
        txt = _norm(lines[idx])
        if txt:
            parts.append(txt)
    return " ".join(parts)


def _window_compact(lines: list[str], start: int, end: int) -> str:
    parts = []
    for idx in range(max(0, start), min(len(lines), end)):
        txt = _compact(lines[idx])
        if txt:
            parts.append(txt)
    return "".join(parts)


def _has_per_capita(lines: list[str], start: int) -> bool:
    return "percapita" in _window_compact(lines, start, start + 3)


def _find_date_idx(lines: list[str], start: int, stop: int) -> int | None:
    for idx in range(start, min(len(lines), stop)):
        if DATE_RE.search(_norm(lines[idx])):
            return idx
    return None


def _collect_nonempty_lines(lines: list[str], start: int, count: int) -> int:
    seen = 0
    idx = start
    while idx < len(lines) and seen < count:
        if _norm(lines[idx]):
            seen += 1
        idx += 1
    return idx


def _numericish(text: str) -> bool:
    text = _norm(text)
    return bool(text) and bool(re.search(r"\d", text))


def _collect_numeric_cluster(lines: list[str], start: int, stop: int) -> tuple[int, int] | None:
    idx = start
    while idx < min(len(lines), stop):
        if not NUMERIC_RE.match(_norm(lines[idx])):
            idx += 1
            continue
        cluster_start = idx
        while idx < min(len(lines), stop) and NUMERIC_RE.match(_norm(lines[idx])):
            idx += 1
        if idx - cluster_start >= 3:
            return cluster_start, idx
    return None


def _rough_value(text: str) -> float | None:
    txt = _norm(text).replace("$", "").replace("r", "").replace("R", "").replace(" ", "")
    if not txt or not re.search(r"\d", txt):
        return None
    if "," in txt and "." in txt:
        if txt.rfind(".") > txt.rfind(","):
            txt = txt.replace(",", "")
        else:
            txt = txt.replace(".", "").replace(",", ".")
    elif "," in txt:
        head, tail = txt.rsplit(",", 1)
        if len(tail) <= 2:
            txt = head.replace(",", "") + "." + tail
        elif len(head) <= 3:
            txt = head + "." + tail
        else:
            txt = txt.replace(",", "")
    elif "." in txt:
        head, tail = txt.rsplit(".", 1)
        if len(tail) > 3:
            txt = txt.replace(".", "")
    try:
        return float(txt)
    except Exception:
        return None


def rule_uscc2_latest_per_capita_currency_and_coin(doc: dict) -> list[dict]:
    try:
        line_objs = doc.get("lines") or []
        if line_objs and isinstance(line_objs[0], dict):
            lines = [str(item.get("text", "")) for item in line_objs]
        else:
            lines = str(doc.get("text", "")).splitlines()
            line_objs = [{"text": text} for text in lines]

        markers = (
            "comparativetotalsofcurrencyandcoinsincirculation",
            "comparativetotalsofcurrencyandcoinincirculation",
            "currencyincirculationbydenomination",
            "amountsoutstandingandincirculation",
            "tableuscc2",
            "uscc2",
        )

        best = None

        for idx in range(len(lines)):
            window = _window_compact(lines, idx, idx + 4)
            if not window:
                continue

            if "percapita" not in window:
                continue

            context = _window_compact(lines, max(0, idx - 8), idx + 1)
            if not any(marker in context for marker in markers):
                continue

            date_idx = _find_date_idx(lines, idx + 1, idx + 16)
            if date_idx is None:
                continue

            header_idx = None
            for back in range(idx, max(-1, idx - 12), -1):
                back_text = _compact(lines[back])
                if "comparativetotalsofcurrencyandcoinsincirculation" in back_text:
                    header_idx = back
                    break
            if header_idx is None:
                for back in range(idx, max(-1, idx - 12), -1):
                    back_text = _compact(lines[back])
                    if (
                        "currencyincirculationbydenomination" in back_text
                        or "tableuscc2" in back_text
                        or "amountsoutstandingandincirculation" in back_text
                    ):
                        header_idx = back
                        break
            if header_idx is None:
                header_idx = max(0, idx - 4)

            end_idx = _collect_nonempty_lines(lines, date_idx, 3)
            span_text = "\n".join(lines[header_idx:end_idx]).strip()
            if not span_text:
                continue

            score = 0
            if "comparativetotalsofcurrencyandcoinsincirculation" in context:
                score += 8
            if "currencyincirculationbydenomination" in context:
                score += 5
            if "tableuscc-2" in context:
                score += 3
            if "amountsoutstandingandincirculation" in context:
                score += 2
            if _has_per_capita(lines, idx):
                score += 2
            score += max(0, 12 - (date_idx - idx))

            candidate = {
                "score": score,
                "header_idx": header_idx,
                "end_idx": end_idx,
                "span_text": span_text,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        if best is None:
            fallback = None
            for idx in range(len(lines)):
                if not DATE_RE.search(_norm(lines[idx])):
                    continue

                end_idx = _collect_nonempty_lines(lines, idx, 3)
                if end_idx <= idx + 1:
                    continue

                row_lines = [line for line in lines[idx:end_idx] if _norm(line)]
                if len(row_lines) < 3:
                    continue
                if not _numericish(row_lines[1]) or not _numericish(row_lines[2]):
                    continue

                context = _window_compact(lines, max(0, idx - 8), idx + 1)
                if not any(
                    marker in context
                    for marker in (
                        "comparativetotalsofcurrencyandcoinsincirculation",
                        "currencyincirculationbydenomination",
                        "amountsoutstandingandincirculation",
                        "totalcurrency",
                        "tableuscc2",
                        "uscc2",
                    )
                ):
                    continue

                score = 0
                if "comparativetotalsofcurrencyandcoinsincirculation" in context:
                    score += 4
                if "currencyincirculationbydenomination" in context:
                    score += 3
                if "totalcurrency" in context:
                    score += 3
                if "tableuscc2" in context or "uscc2" in context:
                    score += 2
                score += max(0, 20 - idx // 100)

                candidate = {
                    "score": score,
                    "header_idx": max(0, idx - 2),
                    "end_idx": end_idx,
                    "span_text": "\n".join(row_lines).strip(),
                }
                if fallback is None or candidate["score"] > fallback["score"]:
                    fallback = candidate

            if fallback is not None:
                best = fallback

        if best is None:
            anchor_idx = None
            for idx in range(len(lines)):
                compact = _window_compact(lines, idx, idx + 3)
                if (
                    "comparativetotalsofcurrencyandcoinsincirculation" in compact
                    or "comparativetotalsofcurrencyandcoinincirculation" in compact
                ):
                    anchor_idx = idx
                    break

            if anchor_idx is None:
                return []

            date_cluster_start = None
            search_stop = min(len(lines), anchor_idx + 450)
            for idx in range(anchor_idx, search_stop):
                if MONTH_RE.search(_norm(lines[idx])) and sum(
                    1 for j in range(idx, min(len(lines), idx + 24)) if MONTH_RE.search(_norm(lines[j]))
                ) >= 5:
                    date_cluster_start = idx
                    break

            if date_cluster_start is None:
                return []

            number_cluster = _collect_numeric_cluster(lines, date_cluster_start, min(len(lines), date_cluster_start + 220))
            if number_cluster is None:
                return []

            cluster_start, cluster_end = number_cluster
            values = [_rough_value(lines[idx]) for idx in range(cluster_start, cluster_end)]
            percap_start = None
            for offset in range(1, len(values)):
                prev = values[offset - 1]
                cur = values[offset]
                if prev is None or cur is None:
                    continue
                if prev > 10000 and cur < 10000:
                    run = values[offset : min(len(values), offset + 5)]
                    if len([v for v in run if v is not None and v < 10000]) >= 4:
                        percap_start = cluster_start + offset
                        break
            if percap_start is None:
                return []

            percap_end = percap_start
            while percap_end < cluster_end:
                value = _rough_value(lines[percap_end])
                if value is None or value >= 10000:
                    break
                percap_end += 1

            best = {
                "header_idx": date_cluster_start,
                "end_idx": min(percap_end, percap_start + 12),
                "span_text": "\n".join(
                    _norm(line)
                    for line in lines[date_cluster_start:min(percap_end, percap_start + 12)]
                    if _norm(line)
                ),
            }

        span = {"text": best["span_text"]}
        if line_objs and isinstance(line_objs[0], dict):
            start_line = line_objs[best["header_idx"]]
            end_line = line_objs[max(best["end_idx"] - 1, best["header_idx"])]
            if start_line.get("page_no") is not None:
                span["page_no"] = start_line.get("page_no")
            if start_line.get("line_no") is not None:
                span["line_no"] = start_line.get("line_no")
            if end_line.get("page_no") is not None:
                span["end_page_no"] = end_line.get("page_no")
            if end_line.get("line_no") is not None:
                span["end_line_no"] = end_line.get("line_no")

        return [span]
    except Exception:
        return []
