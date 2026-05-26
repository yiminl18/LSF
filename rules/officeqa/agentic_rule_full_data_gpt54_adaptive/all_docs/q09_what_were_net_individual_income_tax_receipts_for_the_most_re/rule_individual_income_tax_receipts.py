import re


def rule_individual_income_tax_receipts(doc: dict) -> list[dict]:
    try:
        results = []
        seen = set()

        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def add_span(text: str, meta: dict | None = None) -> None:
            clean = norm(text)
            if not clean:
                return
            key = (
                clean,
                meta.get("page_no") if meta else None,
                meta.get("paragraph_no") if meta else None,
                meta.get("line_no") if meta else None,
            )
            if key in seen:
                return
            seen.add(key)
            span = {"text": clean}
            if meta:
                for field in ("page_no", "paragraph_no", "line_no"):
                    if meta.get(field) is not None:
                        span[field] = meta[field]
            results.append(span)

        def has_individual_income(text: str) -> bool:
            low = (text or "").lower()
            return "individual income" in low or "ndividual income" in low

        sentence_patterns = [
            re.compile(
                r"(?:individual|lndividual) income tax receipts(?:\s*,?\s*net of refunds,?)?\s*were\s*\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*billion[^.]{0,220}",
                re.I,
            ),
            re.compile(
                r"(?:individual|lndividual) income taxes\s*[—–\-.:]+\s*(?:individual|lndividual) income tax receipts(?:\s*,?\s*net of refunds,?)?\s*were\s*\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*billion[^.]{0,220}",
                re.I,
            ),
            re.compile(
                r"(?:individual|lndividual) income taxes.{0,120}?(?:individual|lndividual) income tax receipts(?:\s*,?\s*net of refunds,?)?\s*were\s*\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?\s*billion[^.]{0,220}",
                re.I,
            ),
        ]

        for paragraph in doc.get("paragraphs", []) or []:
            text = paragraph.get("text", "") or ""
            clean = norm(text)
            if not has_individual_income(clean):
                continue
            for pattern in sentence_patterns:
                match = pattern.search(clean)
                if match:
                    add_span(
                        match.group(0),
                        {
                            "page_no": paragraph.get("page_no"),
                            "paragraph_no": paragraph.get("paragraph_no"),
                        },
                    )
                    break

        if results:
            return results

        lines = doc.get("lines", []) or []
        for idx, line in enumerate(lines):
            window_lines = lines[max(0, idx - 1) : min(len(lines), idx + 4)]
            joined = norm(" ".join((item.get("text", "") or "") for item in window_lines))
            if not has_individual_income(joined):
                continue
            for pattern in sentence_patterns:
                match = pattern.search(joined)
                if match:
                    add_span(
                        match.group(0),
                        {
                            "page_no": line.get("page_no"),
                            "line_no": line.get("line_no"),
                        },
                    )
                    break

        if results:
            return results

        text = doc.get("text", "") or ""
        clean_text = norm(text)
        low_text = clean_text.lower()

        if has_individual_income(clean_text):
            for pattern in sentence_patterns:
                match = pattern.search(clean_text)
                if match:
                    add_span(match.group(0))
                    break

        if results:
            return results

        table_title_patterns = [
            re.compile(
                r"(?:first|second|third|fourth).{0,40}(?:quarter|ouar\w{2,6}).{0,40}f(?:iscal|iacal).{0,60}(?:net|nat).{0,20}budg.{0,20}(?:receipts?|racalpta).{0,20}(?:source|sourca).{0,20}\[?in",
                re.I,
            ),
        ]
        number_pattern = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})*|\d+)(?:[.,]\d+)?(?!\w)")

        def norm_number(token: str) -> str:
            clean = token.strip().rstrip(".,;:")
            if "," in clean and "." not in clean and clean.count(",") == 1:
                left, right = clean.split(",", 1)
                if left.isdigit() and right.isdigit() and len(right) <= 3:
                    return f"{left}.{right}"
            return clean

        first_individual = None
        for probe in (
            "individual income taxes",
            "lndividual income taxes",
            "individual income tax receipts",
            "lndividual income tax receipts",
            "individual income",
            "lndividual income",
        ):
            pos = low_text.find(probe)
            if pos != -1:
                first_individual = pos
                break

        for pattern in table_title_patterns:
            for match in pattern.finditer(clean_text):
                start = max(0, match.start() - 80)
                end = min(len(clean_text), match.end() + 2200)
                window = clean_text[start:end]
                if not has_individual_income(window):
                    continue
                low_window = window.lower()
                total_tail = ""
                total_idx = low_window.find("total budget")
                if total_idx != -1:
                    total_tail = window[total_idx:]
                    stop_match = re.search(r"federal fiscal operations|table ffo-1", total_tail, re.I)
                    if stop_match:
                        total_tail = total_tail[: stop_match.start()]
                tail_numbers = number_pattern.findall(total_tail) if total_tail else []
                numbers = number_pattern.findall(window)
                if 0 < len(tail_numbers) <= 2:
                    add_span(norm_number(tail_numbers[-1]))
                elif 0 < len(numbers) <= 2:
                    add_span(norm_number(numbers[-1]))
                else:
                    add_span(window)
                break
            if results:
                break

        if not results and first_individual is not None:
            start = max(0, first_individual - 350)
            end = min(len(clean_text), first_individual + 2200)
            window = clean_text[start:end]
            if any(token in window.lower() for token in ("quarter", "ouart", "budget", "budgat", "[in billions", "source")):
                add_span(window)

        return results
    except Exception:
        return []
