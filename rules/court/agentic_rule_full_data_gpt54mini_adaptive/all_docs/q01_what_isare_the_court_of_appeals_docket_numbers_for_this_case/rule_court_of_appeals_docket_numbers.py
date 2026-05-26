def rule_court_of_appeals_docket_numbers(doc: dict) -> list[dict]:
    try:
        import re

        spans: list[dict] = []
        seen: set[tuple[str, int, int]] = set()

        header_seen = False
        docket_re = re.compile(r"\b\d{2}-\d{3,5}\b")
        prefix_re = re.compile(r"(?i)^\s*Nos?\.\s*(.+?)\s*[\.,;:]*\s*$")
        standalone_re = re.compile(r"^\s*\d{2}-\d{3,5}(?:\s*,\s*\d{2}-\d{3,5})*\s*[\.,;:]*\s*$")
        header_re = re.compile(r"(?i)\b(?:UNITED STATES COURT OF APPEALS|U\.S\. COURT OF APPEALS|COURT OF APPEALS|FOR THE NINTH CIRCUIT)\b")
        stop_re = re.compile(r"(?i)^\s*(?:SUMMARY|COUNSEL|BACKGROUND|FACTS|INTRODUCTION|MEMORANDUM|DISCUSSION)\b")

        for line in doc.get("lines", []):
            page_no = line.get("page_no")
            if isinstance(page_no, int) and page_no > 1:
                break

            text = (line.get("text") or "").strip()
            if not text:
                continue

            if header_re.search(text):
                header_seen = True

            if not header_seen:
                continue

            if stop_re.search(text):
                break

            matches: list[str] = []
            prefix_match = prefix_re.match(text)
            if prefix_match:
                matches.extend(docket_re.findall(prefix_match.group(1)))
            elif standalone_re.match(text):
                matches.extend(docket_re.findall(text))
            else:
                # Some captions place a docket number on a bare line after a
                # "Nos." line. Keep those, but avoid body citations by only
                # accepting lines that are just the docket string.
                if text.startswith("No.") or text.startswith("Nos."):
                    matches.extend(docket_re.findall(text))

            line_no = line.get("line_no")
            for docket in matches:
                key = (docket, page_no if isinstance(page_no, int) else -1, line_no if isinstance(line_no, int) else -1)
                if key in seen:
                    continue
                seen.add(key)
                span = {"text": docket}
                if page_no is not None:
                    span["page_no"] = page_no
                if line_no is not None:
                    span["line_no"] = line_no
                if text:
                    span["context"] = text
                spans.append(span)

        return spans
    except Exception:
        return []
