import re


def rule_ffo4_total_surplus_deficit_fytd(doc: dict) -> list[dict]:
    try:
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", (text or "")).strip()

        def flat(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

        def add_line_span(text: str, line: dict) -> list[dict]:
            snippet = norm(text)
            if not snippet:
                return []
            return [
                {
                    "text": snippet,
                    "page_no": line.get("page_no"),
                    "line_no": line.get("line_no"),
                }
            ]

        def add_paragraph_span(text: str, para: dict) -> list[dict]:
            snippet = norm(text)
            if not snippet:
                return []
            return [
                {
                    "text": snippet,
                    "page_no": para.get("page_no"),
                    "paragraph_no": para.get("paragraph_no"),
                }
            ]

        def add_text_span(text: str) -> list[dict]:
            snippet = norm(text)
            return [{"text": snippet}] if snippet else []

        doc_name = str(doc.get("doc_name", ""))
        year_match = re.search(r"(\d{4})_\d{2}$", doc_name)
        doc_year = year_match.group(1) if year_match else None

        lines = sorted(
            doc.get("lines") or [],
            key=lambda x: (x.get("page_no", 10**9), x.get("line_no", 10**9)),
        )

        if lines:
            lowered = [norm(str(line.get("text", ""))).lower() for line in lines]
            heading_patterns = (
                "total on- and off-budget results and financing of the u.s. government",
                "total on-budget and off-budget results and financing of the u.s. government",
                "total on-budget and oft-budget results",
                "total on-budget and oft-budqet results",
                "total on-budget and off-budget results",
                "total on- and off-budget results",
            )

            for idx, text in enumerate(lowered):
                if not any(pat in text for pat in heading_patterns):
                    continue

                row_idx = None
                search_end = min(len(lines), idx + 40)
                for j in range(idx, search_end):
                    row_text = lowered[j]
                    if "surplus" not in row_text or "deficit" not in row_text:
                        continue
                    if "on-budget" in row_text or "off-budget" in row_text:
                        continue
                    row_idx = j
                    break

                if row_idx is None:
                    continue

                snippet_lines: list[str] = []
                started_numbers = False
                for k in range(row_idx, min(len(lines), row_idx + 6)):
                    raw = norm(str(lines[k].get("text", "")))
                    if not raw:
                        continue
                    snippet_lines.append(raw)
                    if re.search(r"(?<!\w)-?\d[\d,]*", raw):
                        started_numbers = True
                    elif started_numbers and re.search(r"[A-Za-z]", raw):
                        break

                if snippet_lines:
                    numbers = re.findall(r"(?<!\w)-?\d[\d,]*", "\n".join(snippet_lines))
                    if numbers:
                        header_context = " ".join(lowered[max(0, idx - 10) : row_idx + 1])
                        if len(numbers) >= 3 and "budget estimates" in header_context and "full fiscal year" in header_context:
                            if doc_year:
                                fy_pattern = re.compile(
                                    rf"fiscal\s+(?:year\s+)?{doc_year}\s+to\s+date",
                                    re.IGNORECASE,
                                )
                                raw_text = str(doc.get("text", ""))
                                fy_match = fy_pattern.search(raw_text)
                                if fy_match:
                                    start = max(0, fy_match.start() - 2200)
                                    end = min(len(raw_text), fy_match.end() + 1400)
                                    return add_text_span(raw_text[start:end])
                            target = numbers[1]
                        else:
                            target = numbers[1] if len(numbers) >= 2 else numbers[0]
                        return add_line_span(
                            f"Total surplus or deficit (-): {target}",
                            lines[row_idx],
                        )
                    return add_line_span("\n".join(snippet_lines), lines[row_idx])

        raw_text = str(doc.get("text", ""))
        if raw_text:
            summary_table_pattern = re.compile(
                r"table\s+ffo-1.*?summary\s+of\s+f\w+\s+operation\w*",
                re.IGNORECASE | re.DOTALL,
            )
            fiscal_row_patterns = (
                re.compile(r"f\s*i\s*s\s*c\s*a\s*l\s+(?:year\s+)?(?:19|20)\d{2}\s+to\s+date", re.IGNORECASE),
                re.compile(
                    r"a\s*c\s*t\s*u\s*a\s*l\s+f\s*i\s*s\s*c\s*a\s*l\s+y\s*e\s*a\s*r\s+t\s*o\s+d\s*a\s*t\s*e\s+(?:19|20)\d{2}",
                    re.IGNORECASE,
                ),
            )

            for match in summary_table_pattern.finditer(raw_text):
                block_end_match = re.search(
                    r"footnotes\s+to\s+table\s+ffo[-.]?1|table\s+ffo-2",
                    raw_text[match.end() :],
                    re.IGNORECASE,
                )
                block_end = (
                    match.end() + block_end_match.start()
                    if block_end_match
                    else min(len(raw_text), match.end() + 5000)
                )
                block = raw_text[match.start() : block_end]
                flat_block = flat(block)
                if "surplus" not in flat_block or "deficit" not in flat_block:
                    continue
                row_match = None
                for pattern in fiscal_row_patterns:
                    row_match = pattern.search(block)
                    if row_match:
                        break
                if row_match:
                    end = min(len(block), row_match.end() + 1400)
                    return add_text_span(block[:end])

            for pattern in fiscal_row_patterns:
                for match in pattern.finditer(raw_text):
                    start = max(0, match.start() - 4000)
                    end = min(len(raw_text), match.end() + 1200)
                    snippet = raw_text[start:end]
                    flat_snippet = flat(snippet)
                    if "surplus" not in flat_snippet or "deficit" not in flat_snippet:
                        continue
                    if (
                        "summaryoffiscaloperations" in flat_snippet
                        or "totalonbudgetandoffbudgetresults" in flat_snippet
                        or "totalonandoffbudgetresults" in flat_snippet
                        or "budgetandoffbudgetresults" in flat_snippet
                    ):
                        return add_text_span(snippet)

            quarter_summary_pattern = re.compile(
                r"budget\s+results\s+for\s+the\s+fourth\s+quarter\s+and\s+all\s+of\s+fiscal\s+\d{4}",
                re.IGNORECASE,
            )
            for match in quarter_summary_pattern.finditer(raw_text):
                end = min(len(raw_text), match.end() + 5000)
                snippet = raw_text[match.start() : end]
                flat_snippet = flat(snippet)
                if "totalreceipts" in flat_snippet and "surplus" in flat_snippet and "deficit" in flat_snippet:
                    return add_text_span(snippet)

            results_table_pattern = re.compile(
                r"total\s+on[-\s]*and\s+off[-\s]*budget\s+results\s+and\s+financing\s+of\s+the\s+u\.?\s*s\.?\s+government",
                re.IGNORECASE,
            )
            for match in results_table_pattern.finditer(raw_text):
                start = max(0, match.start() - 120)
                end = min(len(raw_text), match.end() + 1500)
                snippet = raw_text[start:end]
                flat_snippet = flat(snippet)
                if "totalreceipts" in flat_snippet and "totalsurplus" in flat_snippet and "deficit" in flat_snippet:
                    return add_text_span(snippet)

        return []
    except Exception:
        return []
