import re
from datetime import datetime


def rule_long_term_debt_snippet(doc: dict) -> list[dict]:
    try:
        lines = sorted(
            doc.get("lines") or [],
            key=lambda x: (x.get("page_no", 10**9), x.get("line_no", 10**9)),
        )

        def clean(text: str) -> str:
            return " ".join(str(text or "").replace("\u00a0", " ").split())

        def nonempty(items: list[dict], start: int, end: int) -> list[str]:
            return [clean(items[i].get("text")) for i in range(start, end) if clean(items[i].get("text"))]

        strong_row_patterns = [
            (re.compile(r"^long[- ]term debt,\s*net$", re.I), 112),
            (re.compile(r"^total net carrying amount$", re.I), 110),
            (re.compile(r"^(?:total\s+)?non-current portion of term debt$", re.I), 108),
            (re.compile(r"^carrying value of long[- ]term debt$", re.I), 107),
            (re.compile(r"^total long[- ]term debt$", re.I), 106),
            (
                re.compile(
                    r"^(?:total\s+)?long[- ]term debt"
                    r"(?:"
                    r",?\s*(?:excluding current portion|less current(?: portion| installments| maturities)?|less:\s*current portion)"
                    r"| and obligations under (?:capital|finance) leases"
                    r"| and finance lease obligations"
                    r")?$",
                    re.I,
                ),
                104,
            ),
            (re.compile(r"^total gross long[- ]term debt$", re.I), 80),
        ]
        bad_row_pattern = re.compile(
            r"(?:current portion of long[- ]term debt|long[- ]term debt due within one year|"
            r"including amounts due within one year|fair value|estimated fair value|"
            r"repayment of long[- ]term debt|repayments of long[- ]term debt|"
            r"proceeds from issuance of long[- ]term debt|cash paid for interest|"
            r"rights of holders|table of contents)",
            re.I,
        )
        debt_note_context = re.compile(
            r"(?:note\s+\d+\s*[.:-]?\s*debt|debt at .* consisted of the following|"
            r"carrying value of (?:our )?(?:borrowings|long[- ]term debt)|"
            r"term debt|long[- ]term debt.*as of|components of long[- ]term debt)",
            re.I,
        )
        header_line_pattern = re.compile(
            r"(?:balance sheets?|statements? of financial position|"
            r"note\s+\d+\s*[.:-]?\s*debt|term debt|debt at .* consisted of the following|"
            r"amounts? in millions|dollars in millions|calendar year|maturities|"
            r"as of|current liabilities|non-current liabilities|long-term liabilities|"
            r"january|february|march|april|may|june|july|august|september|october|november|december|"
            r"\b20\d{2}\b|\b19\d{2}\b)",
            re.I,
        )
        full_date_pattern = re.compile(
            r"^(?:At\s+|As of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}$",
            re.I,
        )
        month_day_pattern = re.compile(
            r"^(?:At\s+|As of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?$",
            re.I,
        )
        year_pattern = re.compile(r"^(19|20)\d{2}$")
        numeric_value_pattern = re.compile(r"^(?:[$€£]\s*)?\(?\d[\d,]*(?:\.\d+)?\)?$")
        skip_context_pattern = re.compile(
            r"(?:fair value hedge|fair value hedges|location on the consolidated balance sheet|carrying value and fair value)",
            re.I,
        )

        by_page: dict[int, list[dict]] = {}
        for line in lines:
            by_page.setdefault(int(line.get("page_no", 1)), []).append(line)

        def extract_values(page_lines: list[dict], idx: int) -> list[str]:
            values: list[str] = []
            for look_ahead in range(idx + 1, min(len(page_lines), idx + 10)):
                value_text = clean(page_lines[look_ahead].get("text"))
                if not value_text or value_text in {"$", "€", "£"}:
                    continue
                if re.fullmatch(r"[()\-–—]+", value_text):
                    values.append("0")
                    continue
                if numeric_value_pattern.match(value_text):
                    normalized = value_text
                    normalized = normalized.lstrip("$€£ ").strip()
                    values.append(normalized)
                    continue
                if re.fullmatch(r"[$€£\s(),.\d\-–—]+", value_text):
                    for match in re.findall(r"\d[\d,]*(?:\.\d+)?", value_text):
                        values.append(match)
                    if values:
                        continue
                if values and re.search(r"[A-Za-z]", value_text):
                    break
            return values[:5]

        def build_date_labels(page_lines: list[dict], idx: int, value_count: int) -> list[str]:
            header_window = [clean(item.get("text")) for item in page_lines[max(0, idx - 100):idx]]
            header_window = [item for item in header_window if item]

            full_dates = [
                re.sub(r"^(?:At|As of)\s+", "", item, flags=re.I).rstrip(",")
                for item in header_window
                if full_date_pattern.match(item)
            ]
            if len(full_dates) >= value_count:
                return full_dates[-value_count:]

            month_days = [
                re.sub(r"^(?:At|As of)\s+", "", item, flags=re.I)
                for item in header_window
                if month_day_pattern.match(item)
            ]
            years = [item for item in header_window if year_pattern.match(item)]
            if month_days and years:
                years = years[-value_count:]
                if len(month_days) == 1:
                    base = month_days[-1].rstrip(",")
                    return [f"{base}, {year}" for year in years]
                if len(month_days) >= value_count and len(years) >= value_count:
                    out: list[str] = []
                    for month_day, year in zip(month_days[-value_count:], years[-value_count:]):
                        out.append(f"{month_day.rstrip(',')}, {year}")
                    return out

            if len(years) >= value_count:
                return years[-value_count:]
            return []

        def parse_label_date(label: str):
            label = label.strip()
            for fmt in ("%B %d, %Y", "%B %d %Y"):
                try:
                    return datetime.strptime(label, fmt)
                except ValueError:
                    pass
            if year_pattern.match(label):
                return datetime(int(label), 12, 31)
            return None

        candidates: list[tuple[int, int, int, str]] = []
        seen_texts: set[str] = set()

        for page_no, page_lines in by_page.items():
            for idx, line in enumerate(page_lines):
                text = clean(line.get("text"))
                low = text.lower()
                if not text:
                    continue
                recent_context = " | ".join(nonempty(page_lines, max(0, idx - 12), idx))
                if low == "long-term debt" and skip_context_pattern.search(recent_context):
                    continue

                score = None
                if not bad_row_pattern.search(text):
                    for pat, pat_score in strong_row_patterns:
                        if pat.match(text):
                            score = pat_score
                            break
                if score is None and low == "debt":
                    prev = " | ".join(nonempty(page_lines, max(0, idx - 3), idx))
                    if re.search(r"(?:long-term liabilities|non-current liabilities)", prev, re.I):
                        score = 92
                if score is None and low == "total debt":
                    prev = " | ".join(nonempty(page_lines, max(0, idx - 20), idx))
                    if debt_note_context.search(prev):
                        score = 78

                if score is None:
                    continue

                values = extract_values(page_lines, idx)
                labels = build_date_labels(page_lines, idx, len(values)) if values else []
                if labels and len(labels) == len(values):
                    score += 5
                elif len(values) > 3:
                    score -= 15

                header_lines: list[str] = []
                for back_idx in range(max(0, idx - 90), idx):
                    header_text = clean(page_lines[back_idx].get("text"))
                    if not header_text:
                        continue
                    if header_line_pattern.search(header_text):
                        if not header_lines or header_lines[-1] != header_text:
                            header_lines.append(header_text)
                header_lines = header_lines[-10:]

                local_start = max(0, idx - 4)
                local_end = min(len(page_lines), idx + 7)
                local_lines = nonempty(page_lines, local_start, local_end)

                if values and labels and len(values) == len(labels):
                    parsed = [parse_label_date(label) for label in labels]
                    if all(item is not None for item in parsed):
                        best_idx = max(range(len(values)), key=lambda i: parsed[i])
                        snippet_lines = [values[best_idx]]
                    else:
                        snippet_lines = [text] + [f"{label}: {value}" for label, value in zip(labels, values)]
                else:
                    snippet_lines = []
                    for snippet_line in header_lines + local_lines:
                        if not snippet_lines or snippet_lines[-1] != snippet_line:
                            snippet_lines.append(snippet_line)

                snippet = "\n".join(snippet_lines).strip()
                if not snippet or snippet in seen_texts:
                    continue
                seen_texts.add(snippet)
                candidates.append((score, page_no, int(line.get("line_no", 10**9)), snippet))

        candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        return [
            {"page_no": page_no, "line_no": line_no, "text": text}
            for _, page_no, line_no, text in candidates[:1]
        ]
    except Exception:
        return []
