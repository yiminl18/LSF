def rule_treasury_bulletin_cover_period(doc: dict) -> list[dict]:
    try:
        import re

        lines = doc.get("lines") or []
        doc_name = (doc.get("doc_name") or "").lower()
        spans = []
        seen = set()

        months = (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
        month_alt = "|".join(months)
        quarter_re = re.compile(r"(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter", re.I)
        season_re = re.compile(r"\b(?:winter|spring|summer|fall|autumn)\s+issue\b", re.I)
        month_year_re = re.compile(rf"\b(?:{month_alt})\s+\d{{4}}\b", re.I)
        month_only_re = re.compile(rf"\b(?:{month_alt})\b", re.I)
        fiscal_year_re = re.compile(r"\bfiscal\s+(?:19|20)?\d{2,4}\b", re.I)
        year_re = re.compile(r"\b(?:19|20)\d{2}\b")
        bulletin_re = re.compile(r"\btreasury\s+bulletin\b", re.I)
        section_heading_re = re.compile(
            r"^(?:INTRODUCTION|CONTENTS|TABLE OF CONTENTS|FINANCIAL OPERATIONS|FEDERAL FISCAL OPERATIONS|"
            r"FEDERAL DEBT|FISCAL SERVICE OPERATIONS|PUBLIC DEBT OPERATIONS|OWNERSHIP OF FEDERAL SECURITIES|"
            r"INTERNATIONAL STATISTICS|CAPITAL MOVEMENTS|U\.S\. SAVINGS|UNITED STATES SAVINGS|"
            r"MARKET BID YIELDS|AVERAGE YIELDS|TREASURY FINANCING)\b",
            re.I,
        )
        doc_name_re = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")
        doc_name_match = doc_name_re.search(doc_name)
        doc_year = int(doc_name_match.group(1)) if doc_name_match else None
        doc_month = int(doc_name_match.group(2)) if doc_name_match else None

        def add_span(text: str, line: dict | None = None) -> None:
            cleaned = " ".join((text or "").split())
            cleaned = re.sub(r"\b(19|20)\s*(\d{2})\b", r"\1\2", cleaned)
            cleaned = re.sub(r"\b1\s*(\d{3})\b", r"1\1", cleaned)
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            span = {"text": cleaned}
            if line is not None:
                if line.get("page_no") is not None:
                    span["page_no"] = line.get("page_no")
                if line.get("line_no") is not None:
                    span["line_no"] = line.get("line_no")
            spans.append(span)

        def best_month_year_from_doc_name() -> str | None:
            if doc_year is None or doc_month is None:
                return None
            month_name = months[doc_month - 1]
            return f"{month_name} {doc_year}"

        def quarter_from_month(month: int) -> str:
            if month in (1, 2, 3):
                return "First quarter"
            if month in (4, 5, 6):
                return "Second quarter"
            if month in (7, 8, 9):
                return "Third quarter"
            return "Fourth quarter"

        # Prioritize the most explicit cover phrasing.
        search_lines = lines[:120]
        explicit_matches = []
        month_matches = []
        season_matches = []
        fiscal_matches = []

        for line in search_lines:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if quarter_re.search(text):
                explicit_matches.append((0, line, text))
                continue
            if season_re.search(text):
                season_matches.append((1, line, text))
                continue
            if month_year_re.search(text):
                month_matches.append((2, line, text))
                continue
            if bulletin_re.search(text) and year_re.search(text):
                explicit_matches.append((3, line, text))
                continue
            if fiscal_year_re.search(text):
                fiscal_matches.append((4, line, text))

        if explicit_matches:
            for _, line, text in explicit_matches:
                add_span(text, line)
            return spans

        if season_matches:
            # Keep the season line and, when present, the nearest fiscal year line.
            for _, line, text in season_matches:
                add_span(text, line)
                line_no = line.get("line_no")
                page_no = line.get("page_no")
                if line_no is not None:
                    combined_parts = [text]
                    for other in search_lines:
                        if other.get("page_no") != page_no:
                            continue
                        other_line_no = other.get("line_no")
                        if other_line_no is None:
                            continue
                        if 0 <= other_line_no - line_no <= 4:
                            other_text = (other.get("text") or "").strip()
                            if (
                                fiscal_year_re.search(other_text)
                                or year_re.fullmatch(other_text)
                                or year_re.search(other_text)
                                or other_text.lower() == "fiscal"
                                or re.fullmatch(r"\d[\d\s]{2,5}", other_text)
                            ):
                                combined_parts.append(other_text)
                    if len(combined_parts) > 1:
                        add_span(" ".join(combined_parts), line)
            return spans

        if month_matches:
            for _, line, text in month_matches:
                add_span(text, line)
            return spans

        # Handle month-only OCR hits by pairing them with the file name year.
        for line in search_lines:
            text = (line.get("text") or "").strip()
            if not text:
                continue
            if month_only_re.search(text):
                month_name = month_only_re.search(text).group(0)
                if year_re.search(text):
                    add_span(text, line)
                elif doc_year is not None:
                    add_span(f"{month_name} {doc_year}", line)

        if spans:
            return spans

        # Last resort: derive a normalized period from the filename.
        if doc_month is not None and doc_year is not None:
            add_span(best_month_year_from_doc_name() or "")
            return spans

        return []
    except Exception:
        return []
