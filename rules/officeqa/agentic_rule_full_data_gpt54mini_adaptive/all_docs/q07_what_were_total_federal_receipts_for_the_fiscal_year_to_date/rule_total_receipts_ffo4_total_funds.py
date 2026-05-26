import re


def rule_total_receipts_ffo4_total_funds(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return " ".join((text or "").split())

        def parse_number(text: str) -> int | None:
            cleaned = text.replace(".", ",")
            try:
                return int(cleaned.replace(",", ""))
            except Exception:
                return None

        def extract_summary_value(lines: list[dict]) -> tuple[str, dict] | None:
            heading_pat = re.compile(
                r"total\s+on[-\s]*and\s+off[-\s]*budget\s+results"
                r"(?:\s+and\s+financing\s+of\s+the\s+u\.s\.\s+government)?",
                re.IGNORECASE,
            )
            row_pat = re.compile(r"^total\s+receipts\b", re.IGNORECASE)
            num_pat = re.compile(r"\d{1,3}(?:[,.]\d{3})+")

            for idx, item in enumerate(lines):
                if not heading_pat.search(norm(item.get("text") or "")):
                    continue

                row_idx = None
                for j in range(idx, min(len(lines), idx + 30)):
                    if row_pat.match(norm(lines[j].get("text") or "")):
                        row_idx = j
                        break
                if row_idx is None:
                    continue

                numbers: list[str] = []
                started = False
                for k in range(row_idx, min(len(lines), row_idx + 12)):
                    raw = norm(lines[k].get("text") or "")
                    found = num_pat.findall(raw)
                    if found:
                        started = True
                        numbers.extend(found)
                        continue
                    if started and re.search(r"[A-Za-z]", raw):
                        break

                if numbers:
                    row_line = lines[row_idx]
                    return numbers[-1].replace(".", ","), row_line
            return None

        def extract_detail_value(lines: list[dict]) -> tuple[str, dict] | None:
            anchor_pat = re.compile(r"^budget\s+receipts:\s*$", re.IGNORECASE)
            row_pat = re.compile(r"^total\s+receipts\b", re.IGNORECASE)
            num_pat = re.compile(r"\d{1,3}(?:[,.]\d{3})+")

            for idx, item in enumerate(lines):
                if not anchor_pat.match(norm(item.get("text") or "")):
                    continue

                row_idx = None
                for j in range(idx, min(len(lines), idx + 140)):
                    if row_pat.match(norm(lines[j].get("text") or "")):
                        row_idx = j
                        break
                if row_idx is None:
                    continue

                numbers: list[str] = []
                started = False
                for k in range(row_idx, min(len(lines), row_idx + 16)):
                    raw = norm(lines[k].get("text") or "")
                    found = num_pat.findall(raw)
                    if found:
                        started = True
                        numbers.extend(found)
                        continue
                    if started and re.search(r"[A-Za-z]", raw):
                        break

                if len(numbers) >= 4:
                    row_line = lines[row_idx]
                    return numbers[3].replace(".", ","), row_line
            return None

        lines = sorted(
            doc.get("lines") or [],
            key=lambda x: (
                x.get("page_no", 10**9),
                x.get("line_no", 10**9),
            ),
        )

        summary = extract_summary_value(lines)
        detail = extract_detail_value(lines)

        chosen: tuple[str, dict] | None = None
        if summary and detail:
            summary_num = parse_number(summary[0])
            detail_num = parse_number(detail[0])
            if summary_num is not None and detail_num is not None and abs(summary_num - detail_num) <= 1:
                chosen = detail
            else:
                chosen = summary
        else:
            chosen = detail or summary

        if chosen:
            value, row_line = chosen
            return [
                {
                    "text": f"Total federal receipts for the fiscal year to date were {value}.",
                    "page_no": row_line.get("page_no"),
                    "line_no": row_line.get("line_no"),
                }
            ]

        return []
    except Exception:
        return []
