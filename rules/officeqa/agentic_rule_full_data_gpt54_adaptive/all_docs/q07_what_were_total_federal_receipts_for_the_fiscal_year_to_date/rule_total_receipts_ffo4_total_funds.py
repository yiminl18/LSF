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

        def doc_year() -> int | None:
            match = re.search(r"(\d{4})_(\d{2})$", str(doc.get("doc_name") or ""))
            if not match:
                return None
            try:
                return int(match.group(1))
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

                header_blob = " ".join(
                    norm(lines[k].get("text") or "")
                    for k in range(idx, min(len(lines), row_idx + 1))
                )
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
                    if len(numbers) == 1:
                        chosen = numbers[0]
                    elif re.search(r"\bfirst\s+quarter\b", header_blob, re.IGNORECASE):
                        chosen = numbers[0]
                    else:
                        chosen = numbers[-1]
                    return chosen.replace(".", ","), row_line
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

        def extract_legacy_fiscal_block_value(lines: list[dict]) -> tuple[str, dict] | None:
            anchor_pat = re.compile(
                r"total\s+on[-\s]*budget\s+and\s+off[-\s]*budget\s+financing",
                re.IGNORECASE,
            )
            num_pat = re.compile(r"\$?\d{1,3}(?:[,.]\d{3})+")

            for idx, item in enumerate(lines):
                if not anchor_pat.search(norm(item.get("text") or "")):
                    continue

                dollar_hits: list[tuple[str, dict]] = []
                for j in range(idx + 1, min(len(lines), idx + 25)):
                    raw = norm(lines[j].get("text") or "")
                    if re.match(r"^\$\d", raw):
                        cleaned = re.sub(r"[^0-9,.\-]", "", raw.lstrip("$"))
                        if cleaned:
                            dollar_hits.append((cleaned.replace(".", ","), lines[j]))
                if dollar_hits:
                    return dollar_hits[-1]

                for j in range(idx + 1, min(len(lines), idx + 8)):
                    raw = norm(lines[j].get("text") or "")
                    if not raw:
                        continue
                    if num_pat.fullmatch(raw):
                        continue
                    if re.search(r"[A-Za-z]", raw) and re.search(r"\d", raw):
                        for k in range(j + 1, min(len(lines), j + 4)):
                            found = num_pat.findall(norm(lines[k].get("text") or ""))
                            if found:
                                return found[0].replace(".", ","), lines[k]

                label_idx = None
                for j in range(idx + 1, min(len(lines), idx + 6)):
                    raw = norm(lines[j].get("text") or "")
                    if raw and re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", raw, re.IGNORECASE):
                        label_idx = j
                        break

                if label_idx is None:
                    continue

                numbers: list[str] = []
                number_lines: list[dict] = []
                started = False
                for k in range(label_idx + 1, min(len(lines), label_idx + 40)):
                    raw = norm(lines[k].get("text") or "")
                    found = num_pat.findall(raw)
                    if found:
                        started = True
                        for value in found:
                            numbers.append(value)
                            number_lines.append(lines[k])
                        continue
                    if started and re.search(r"[A-Za-z]", raw):
                        break

                if len(numbers) >= 14:
                    return numbers[13].replace(".", ","), number_lines[13]
            return None

        def extract_early_to_date_value(lines: list[dict]) -> tuple[str, dict] | None:
            year = doc_year()
            if year is None or year < 1982 or year > 1987:
                return None

            num_pat = re.compile(r"\$?\d{1,3}(?:[,.]\d{3})+")

            for idx in range(len(lines)):
                joined = " ".join(
                    norm(lines[j].get("text") or "")
                    for j in range(idx, min(len(lines), idx + 4))
                )
                if not re.search(rf"\bfiscal\b.*\b{year}\b.*\bto\s+date\b", joined, re.IGNORECASE):
                    continue

                start_k = idx
                for j in range(idx, min(len(lines), idx + 4)):
                    if re.search(r"\bto\s+date\b", norm(lines[j].get("text") or ""), re.IGNORECASE):
                        start_k = j + 1
                        break

                for k in range(start_k, min(len(lines), start_k + 8)):
                    found = num_pat.findall(norm(lines[k].get("text") or ""))
                    if found:
                        return found[0].replace(".", ","), lines[k]
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
        legacy = extract_legacy_fiscal_block_value(lines)
        early = extract_early_to_date_value(lines)

        chosen: tuple[str, dict] | None = None
        if summary and detail:
            summary_num = parse_number(summary[0])
            detail_num = parse_number(detail[0])
            if summary_num is not None and detail_num is not None and abs(summary_num - detail_num) <= 1:
                chosen = summary if summary_num <= detail_num else detail
            else:
                chosen = summary
        else:
            chosen = summary or detail or legacy or early

        if chosen is None:
            chosen = legacy or early

        if chosen:
            value, row_line = chosen
            return [
                {
                    "text": value.lstrip("$"),
                    "page_no": row_line.get("page_no"),
                    "line_no": row_line.get("line_no"),
                }
            ]

        return []
    except Exception:
        return []
