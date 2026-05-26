import re


_MONTH_RE = re.compile(
    r"\b("
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|"
    r"october|november|december"
    r")\b",
    re.IGNORECASE,
)


def rule_fd1_latest_fy_public_total(doc: dict) -> list[dict]:
    try:
        doc_match = re.search(r"_(\d{4})_(\d{2})$", str(doc.get("doc_name", "")))
        doc_year = int(doc_match.group(1)) if doc_match else None
        doc_month = int(doc_match.group(2)) if doc_match else None

        line_objs = [item for item in (doc.get("lines") or []) if isinstance(item, dict)]
        if not line_objs:
            return []

        def norm(text: str) -> str:
            text = str(text or "")
            for old, new in (("—", "-"), ("–", "-"), ("−", "-"), ("•", " ")):
                text = text.replace(old, new)
            return re.sub(r"\s+", " ", text).strip()

        def digit_year(text: str) -> str | None:
            digits = "".join(ch for ch in text if ch.isdigit())
            if len(digits) != 4:
                return None
            if not (digits.startswith("19") or digits.startswith("20")):
                return None
            return digits

        def is_annual_label(text: str) -> bool:
            cleaned = norm(text)
            if not cleaned or _MONTH_RE.search(cleaned):
                return False
            if "t.q" in cleaned.lower() or "t.o" in cleaned.lower():
                return False
            return digit_year(cleaned) is not None

        def is_month_label(text: str) -> bool:
            cleaned = norm(text)
            if not cleaned:
                return False
            lowered = cleaned.lower()
            if "t.q" in lowered or "t.o" in lowered:
                return True
            if _MONTH_RE.search(cleaned):
                return True
            if digit_year(cleaned) and "-" in cleaned:
                return True
            return False

        def is_split_month_row(idx: int) -> bool:
            current = norm(line_objs[idx].get("text", ""))
            if digit_year(current) is None:
                return False
            lookahead = " ".join(
                norm(line_objs[j].get("text", ""))
                for j in range(idx, min(len(line_objs), idx + 3))
            )
            return "-" in lookahead and _MONTH_RE.search(lookahead) is not None

        def is_numeric(text: str) -> bool:
            cleaned = norm(text).replace("$", "")
            if not cleaned or any(ch.isalpha() for ch in cleaned):
                return False
            if cleaned in {"-", "--"}:
                return True
            digits = sum(ch.isdigit() for ch in cleaned)
            if digits == 0:
                return False
            return all(ch.isdigit() or ch in {",", ".", "-", " "} for ch in cleaned)

        def add_span(spans: list[dict], seen: set[str], text: str, start_idx: int) -> None:
            cleaned = "\n".join(norm(part) for part in text.splitlines() if norm(part))
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            source = line_objs[start_idx]
            span = {"text": cleaned}
            if source.get("page_no") is not None:
                span["page_no"] = source.get("page_no")
            if source.get("line_no") is not None:
                span["line_no"] = source.get("line_no")
            spans.append(span)

        starts: list[int] = []
        for i in range(len(line_objs)):
            window = " ".join(norm(line_objs[j].get("text", "")) for j in range(i, min(len(line_objs), i + 4))).lower()
            if "summary of federal debt" not in window:
                continue
            if "table fd-1" not in window and "table fd-l" not in window:
                continue
            nearby = " ".join(
                norm(line_objs[j].get("text", ""))
                for j in range(i, min(len(line_objs), i + 60))
            ).lower()
            if "amount outstanding" not in nearby:
                continue
            if "the public" not in nearby and "securities held by" not in nearby:
                continue
            starts.append(i)

        spans: list[dict] = []
        seen: set[str] = set()

        for start in starts:
            scan_end = min(len(line_objs), start + 320)
            annual_rows: list[int] = []
            for i in range(start, scan_end):
                text = norm(line_objs[i].get("text", ""))
                if not text:
                    continue
                if i > start + 3 and text.lower().startswith("table fd"):
                    break
                if is_split_month_row(i):
                    if annual_rows:
                        break
                    continue
                if is_annual_label(text):
                    annual_rows.append(i)
                    continue
                if annual_rows and is_month_label(text):
                    break

            if not annual_rows:
                continue

            row_start = annual_rows[-1]
            row_end = min(scan_end, row_start + 40)
            for i in range(row_start + 1, scan_end):
                text = norm(line_objs[i].get("text", ""))
                if not text:
                    continue
                if is_split_month_row(i) or is_annual_label(text) or (i > row_start + 1 and is_month_label(text)) or text.lower().startswith("table fd"):
                    row_end = i
                    break

            numeric_items = [
                (i, norm(line_objs[i].get("text", "")))
                for i in range(row_start + 1, row_end)
                if is_numeric(line_objs[i].get("text", ""))
            ]
            if len(numeric_items) < 7:
                continue

            target_value = numeric_items[6][1].replace("$", "")
            if doc_year is not None and doc_month is not None:
                expected_year = doc_year if doc_month >= 10 else doc_year - 1
                add_span(spans, seen, f"{expected_year} The public total {target_value}", row_start)
            else:
                year_text = digit_year(norm(line_objs[row_start].get("text", ""))) or norm(line_objs[row_start].get("text", ""))
                add_span(spans, seen, f"{year_text} The public total {target_value}", row_start)

            header_start = max(start, row_start - 26)
            context_end = min(scan_end, max(row_end, row_start + 12))
            context_text = "\n".join(
                str(line_objs[i].get("text", ""))
                for i in range(header_start, context_end)
            )
            add_span(spans, seen, context_text, header_start)

        return spans
    except Exception:
        return []
