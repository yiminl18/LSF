def rule_court_of_appeals_docket_numbers(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        spans: list[dict] = []
        seen: set[tuple[int, int, str]] = set()

        docket_re = re.compile(r"\b\d{1,2}(?:A)?-\d{1,6}(?:-\d{1,6})?\b")
        docket_only_re = re.compile(r"^\s*\d{1,2}(?:A)?-\d{1,6}(?:-\d{1,6})?\s*$")
        header_re = re.compile(
            r"(?i)\b(?:UNITED STATES COURT OF APPEALS|U\.S\.\s*COURT OF APPEALS|COURT OF APPEALS)\b"
        )
        label_inline_re = re.compile(r"(?i)^\s*(Nos?)\.?\s*(.*)$")
        stop_re = re.compile(
            r"(?i)^\s*(?:Appeal from|On Appeal from|On Petition for Review|On Petition for Writ|"
            r"On Petition to|Argued and Submitted|Submitted|Before:|COUNSEL|SUMMARY\b)"
        )
        judge_re = re.compile(r"^(?:PER CURIAM|[A-Z][A-Za-z .'-]+,\s*(?:Chief\s+)?(?:Circuit\s+)?Judge:)")
        lower_court_re = re.compile(
            r"(?i)^\s*(?:D\.C\.|District Court|Agency|BAP|Bankr(?:uptcy)?|Bk\.?|Tax Court|"
            r"Court of International Trade|CIT)\s+No"
        )

        header_seen = False
        pending_mode: str | None = None
        pending_gap = 0

        def add_matches(text: str, line: dict) -> None:
            page_no = int(line.get("page_no") or -1)
            line_no = int(line.get("line_no") or -1)
            for match in docket_re.finditer(text):
                value = match.group(0).rstrip(".,;:")
                key = (page_no, line_no, value)
                if key in seen:
                    continue
                seen.add(key)
                span = {"text": value}
                if line.get("page_no") is not None:
                    span["page_no"] = line["page_no"]
                if line.get("line_no") is not None:
                    span["line_no"] = line["line_no"]
                span["context"] = (line.get("text") or "").strip()
                spans.append(span)

        for line in lines:
            page_no = line.get("page_no")
            if isinstance(page_no, int) and page_no > 8:
                break

            raw_text = line.get("text") or ""
            text = raw_text.strip()
            if not text:
                continue

            if header_re.search(text):
                header_seen = True

            if not header_seen:
                continue

            if spans and (stop_re.search(text) or judge_re.search(text)):
                break

            if lower_court_re.match(text):
                pending_mode = None
                pending_gap = 0
                continue

            inline_match = label_inline_re.match(text)
            if inline_match:
                label = inline_match.group(1).lower()
                remainder = inline_match.group(2).strip()
                if remainder:
                    add_matches(remainder, line)
                pending_mode = "plural" if label.startswith("nos") else ("single" if not remainder else None)
                pending_gap = 0
                if label.startswith("nos") and not remainder:
                    pending_mode = "plural"
                continue

            if pending_mode and docket_only_re.match(text):
                add_matches(text, line)
                pending_gap = 0
                if pending_mode == "single":
                    pending_mode = None
                continue

            if pending_mode:
                if header_re.search(text) or re.fullmatch(r"\d+", text):
                    pending_gap += 1
                    if pending_gap <= 2:
                        continue
                pending_mode = None
                pending_gap = 0

        return spans
    except Exception:
        return []
