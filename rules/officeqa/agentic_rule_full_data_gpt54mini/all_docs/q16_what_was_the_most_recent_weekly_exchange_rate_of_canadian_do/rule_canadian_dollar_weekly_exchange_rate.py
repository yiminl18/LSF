import re


_WEEKLY_TITLE_RE = re.compile(
    r"\bWeekly\b.*\b(?:Report of Major Market Participants|Bank Positions)\b",
    re.IGNORECASE,
)
_CANADIAN_CONTEXT_RE = re.compile(
    r"\bCanadian\s+Dollar\s+Positions\b|\bCanadian\s+dollar\b|\bCanadian dollars per U\.S\. dollar\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2}\b")
_NUMERIC_RE = re.compile(r"^[rR]?-?\d[\d,]*(?:\.\d+)?$")
_END_TITLE_RE = re.compile(
    r"\b(?:TABLE|Table)\b.*\b(?:Monthly|Quarterly|Nonbanking|Consolidated)\b",
    re.IGNORECASE,
)


def rule_canadian_dollar_weekly_exchange_rate(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def is_numeric_candidate(text: str) -> bool:
            t = norm(text).replace("$", "")
            return bool(_NUMERIC_RE.match(t) or t.lower() in {"n.a.", "na"})

        def row_exchange_rate(row: list[dict]) -> dict | None:
            if not row:
                return None
            numeric_items = [item for item in row if is_numeric_candidate(item.get("text") or "")]
            if not numeric_items:
                return None
            preferred = None
            for item in reversed(numeric_items):
                if "." in norm(item.get("text") or ""):
                    preferred = item
                    break
            chosen = preferred or numeric_items[-1]
            span = {"text": norm(chosen.get("text") or "")}
            if chosen.get("page_no") is not None:
                span["page_no"] = chosen.get("page_no")
            if chosen.get("line_no") is not None:
                span["line_no"] = chosen.get("line_no")
            if chosen.get("paragraph_no") is not None:
                span["paragraph_no"] = chosen.get("paragraph_no")
            return span

        def parse_block(block_items: list[dict]) -> list[dict]:
            rows: list[list[dict]] = []
            current: list[dict] = []
            saw_date = False

            for item in block_items:
                text = norm(item.get("text") or "")
                if not text:
                    continue
                if _END_TITLE_RE.search(text) and rows:
                    break
                if _WEEKLY_TITLE_RE.search(text) or _CANADIAN_CONTEXT_RE.search(text):
                    if current and saw_date:
                        # Keep scanning; titles can repeat on continued pages.
                        pass
                    continue
                if _DATE_RE.search(text):
                    if current and saw_date:
                        rows.append(current)
                    current = [item]
                    saw_date = True
                    continue
                if current and is_numeric_candidate(text):
                    current.append(item)
                    continue
            if current and saw_date:
                rows.append(current)

            if not rows:
                return []
            answer = row_exchange_rate(rows[-1])
            return [answer] if answer else []

        paragraphs = [p for p in (doc.get("paragraphs") or []) if isinstance(p, dict)]
        for para in paragraphs:
            text = para.get("text") or ""
            raw_lines = text.splitlines()
            if not raw_lines:
                continue
            title_line_idx = None
            for idx, line in enumerate(raw_lines):
                line_text = norm(line)
                if not line_text:
                    continue
                if _WEEKLY_TITLE_RE.search(line_text) and _CANADIAN_CONTEXT_RE.search(
                    " ".join(norm(part) for part in raw_lines[max(0, idx - 4) : idx + 1])
                ):
                    title_line_idx = idx
                    break
            if title_line_idx is None:
                continue
            trailing_text = "\n".join(raw_lines[title_line_idx + 1 :])
            if not _DATE_RE.search(trailing_text):
                continue
            items = []
            for idx, line in enumerate(raw_lines[title_line_idx + 1 :], start=title_line_idx + 2):
                items.append(
                    {
                        "text": line,
                        "page_no": para.get("page_no"),
                        "paragraph_no": para.get("paragraph_no"),
                        "line_no": idx,
                    }
                )
            spans = parse_block(items)
            if spans:
                return spans

        lines = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not lines:
            return []

        candidate_starts: list[int] = []
        for idx, item in enumerate(lines):
            text = norm(item.get("text") or "")
            if not text:
                continue
            if not _WEEKLY_TITLE_RE.search(text):
                continue
            context = " ".join(norm(lines[j].get("text") or "") for j in range(max(0, idx - 4), idx))
            if not _CANADIAN_CONTEXT_RE.search(context) and not _CANADIAN_CONTEXT_RE.search(text):
                continue
            candidate_starts.append(idx)

        for start_idx in candidate_starts:
            block: list[dict] = []
            for item in lines[start_idx : min(len(lines), start_idx + 600)]:
                block.append(item)
                text = norm(item.get("text") or "")
                if (
                    block
                    and _END_TITLE_RE.search(text)
                    and any(_DATE_RE.search(norm(prev.get("text") or "")) for prev in block)
                ):
                    break
            spans = parse_block(block)
            if spans:
                return spans

        return []
    except Exception:
        return []
