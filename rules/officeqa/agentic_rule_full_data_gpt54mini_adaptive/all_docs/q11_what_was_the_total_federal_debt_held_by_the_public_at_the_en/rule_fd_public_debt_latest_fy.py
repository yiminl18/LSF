import re


def rule_fd_public_debt_latest_fy(doc: dict) -> list[dict]:
    try:
        line_objs = doc.get("lines") or []
        if line_objs and isinstance(line_objs[0], dict):
            lines = [str(x.get("text", "")) for x in line_objs]
        else:
            lines = str(doc.get("text", "")).splitlines()
            line_objs = [{"text": t} for t in lines]

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", str(text or "")).strip()

        month_pat = re.compile(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
            r"january|february|march|april|june|july|august|september|"
            r"october|november|december)\b",
            re.I,
        )
        year_pat = re.compile(r"^\s*(19|20)\d{2}\b")

        def find_start(phrases: tuple[str, ...]) -> int | None:
            for i in range(len(lines)):
                window = " ".join(norm(lines[j]) for j in range(i, min(len(lines), i + 3))).lower()
                if "table fd" not in window:
                    continue
                if not any(p in window for p in phrases):
                    continue
                if any(
                    year_pat.match(norm(lines[j])) for j in range(i + 1, min(len(lines), i + 120))
                ):
                    return i
            for i in range(len(lines) - 1):
                window = (norm(lines[i]) + " " + norm(lines[i + 1])).lower()
                if "table fd" not in window:
                    continue
                if not any(p in window for p in phrases):
                    continue
                if any(
                    year_pat.match(norm(lines[j])) for j in range(i + 1, min(len(lines), i + 120))
                ):
                    return i
            return None

        def extract_latest_row(start: int) -> dict | None:
            search_end = min(len(lines), start + 260)
            year_rows: list[int] = []
            saw_year_block = False
            for i in range(start, search_end):
                t = norm(lines[i])
                if not t:
                    continue
                if year_pat.match(t) and not month_pat.search(t):
                    year_rows.append(i)
                    saw_year_block = True
                    continue
                if saw_year_block:
                    if month_pat.search(t):
                        break
                    if t.lower().startswith("table fd"):
                        break
            if not year_rows:
                return None
            row_start = year_rows[-1]
            row_end = min(search_end, row_start + 30)
            for i in range(row_start + 1, row_end):
                if norm(lines[i]).lower().startswith("table fd"):
                    row_end = i
                    break

            span_text = "\n".join(lines[start:row_end]).strip()
            if not span_text:
                return None

            span = {"text": span_text}
            if line_objs and isinstance(line_objs[row_start], dict):
                span["page_no"] = line_objs[row_start].get("page_no")
                span["line_no"] = line_objs[row_start].get("line_no")
                span["end_page_no"] = line_objs[row_end - 1].get("page_no")
                span["end_line_no"] = line_objs[row_end - 1].get("line_no")
            return span

        spans: list[dict] = []

        # First try the modern summary table, then fall back to the older public-debt table.
        for phrases in (
            ("summary of federal debt",),
            ("debt held by the public", "interest-bearing public debt"),
        ):
            start = find_start(phrases)
            if start is None:
                continue
            span = extract_latest_row(start)
            if span is not None:
                spans.append(span)

        return spans
    except Exception:
        return []
