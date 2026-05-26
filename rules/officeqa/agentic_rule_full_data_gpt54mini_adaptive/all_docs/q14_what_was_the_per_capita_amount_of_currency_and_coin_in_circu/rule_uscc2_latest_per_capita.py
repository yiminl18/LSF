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


def rule_uscc2_latest_per_capita(doc: dict) -> list[dict]:
    try:
        line_objs = doc.get("lines") or []
        if line_objs and isinstance(line_objs[0], dict):
            lines = [str(item.get("text", "")) for item in line_objs]
        else:
            lines = str(doc.get("text", "")).splitlines()
            line_objs = [{"text": text} for text in lines]

        markers = (
            "comparativetotalsofcurrencyandcoinsincirculation",
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

            if fallback is None:
                return []
            best = fallback

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
