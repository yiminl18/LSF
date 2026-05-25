import re


def rule_long_term_debt_line_window(doc: dict) -> list[dict]:
    try:
        out = []
        seen = set()
        kw_re = re.compile(
            r"("
            r"long[- ]term debt and obligations under finance leases|"
            r"long[- ]term debt and finance lease obligations|"
            r"long[- ]term debt and obligations|"
            r"long[- ]term debt obligations|"
            r"long[- ]term debt due within one year|"
            r"current portion of (?:debt|long[- ]term debt)(?: and obligations under finance leases)?|"
            r"debt and obligations under finance leases|"
            r"carrying value of long[- ]term debt|"
            r"long[- ]term obligations|"
            r"long[- ]term debt|"
            r"total debt outstanding|"
            r"debt and borrowings"
            r")",
            re.IGNORECASE,
        )
        row_start_re = re.compile(
            r"^\s*(?:"
            r"long[- ]term debt(?:\b|,|$| and obligations under finance leases| and finance lease obligations|"
            r" and obligations| obligations| due within one year|, including amounts due within one year)|"
            r"current portion of (?:debt|long[- ]term debt)(?: and obligations under finance leases)?|"
            r"debt and obligations under finance leases|"
            r"carrying value of long[- ]term debt|"
            r"long[- ]term obligations|"
            r"total debt outstanding"
            r")",
            re.IGNORECASE,
        )
        short_row_re = re.compile(r"^\s*(?:\$|[()\d,.\-]+|\s)+\s*$")
        date_header_re = re.compile(
            r"(?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b20\d{2}\b|\b19\d{2}\b|"
            r"december\s+31,|june\s+30,|april\s+30,|january\s+31,|may\s+31,|march\s+31,|"
            r"september\s+30,|august\s+31,|february\s+28,|february\s+29,|year ended)",
            re.IGNORECASE,
        )
        continuation_re = re.compile(
            r"^(?:within one year|due within one year|current portion(?: of(?: long[- ]term)? debt)?|"
            r"thereafter|less: current portion.*|total|balances as of .+)$",
            re.IGNORECASE,
        )

        def alpha_word_count(text: str) -> int:
            return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))

        def is_row_like(text: str) -> bool:
            low = text.lower().strip()
            if not low:
                return False
            if row_start_re.search(low):
                return alpha_word_count(low) <= 12
            return False

        def is_context_line(text: str) -> bool:
            low = text.lower().strip()
            if not low:
                return False
            if short_row_re.match(low):
                return True
            if re.fullmatch(r"[\d,().$\-\s]+", low):
                return True
            if date_header_re.search(low):
                return alpha_word_count(low) <= 12
            if continuation_re.match(low):
                return True
            return False

        by_page = {}
        for line in doc.get("lines") or []:
            by_page.setdefault(line.get("page_no"), []).append(line)

        for page_no, page_lines in by_page.items():
            for idx, line in enumerate(page_lines):
                text = str(line.get("text") or "").strip()
                if not text or not is_row_like(text):
                    continue

                start = max(0, idx - 12)
                end = min(len(page_lines), idx + 4)
                snippet_lines = []
                for item in page_lines[start:end]:
                    item_text = str(item.get("text") or "").strip()
                    if item_text and is_context_line(item_text):
                        snippet_lines.append(item_text)

                snippet = "\n".join(snippet_lines).strip()
                if not snippet:
                    continue

                key = (page_no, line.get("line_no"), snippet)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "page_no": page_no,
                        "line_no": line.get("line_no"),
                        "text": snippet,
                    }
                )

        return out
    except Exception:
        return []
